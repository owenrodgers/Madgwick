#![no_std]
#![no_main]

mod quaternion;
mod madgwick;

use defmt_rtt as _;
use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, channel::Channel};
use lsm303agr::{Lsm303agr, MagMode, MagOutputDataRate, interface::I2cInterface, mode::MagContinuous};
use mpu6050_async::Mpu6050;
use panic_probe as _;
use postcard::to_slice;
use static_cell::ConstStaticCell;
use embassy_executor::Spawner;
use embassy_time::{Delay, Duration, Ticker, WithTimeout};
use embassy_nrf::{bind_interrupts, peripherals, twim::{self, Twim}, uarte::{self}};
use serde::{Deserialize, Serialize};
use zerocopy::{IntoBytes, Immutable, KnownLayout};
use madgwick::MadgwickFilter;

use crate::madgwick::Madgwick;

const UART_FRAME_START : u32 = 0x77_77_77_77;

/* 
    Measured sensor biases from sensor-bench
*/
const GYRO_X_BIAS : f32 = -0.002873; 
const GYRO_Y_BIAS : f32 = 0.040268;
const GYRO_Z_BIAS : f32 = 0.004818;

const ACC_X_BIAS: f32 = 0.102714;
const ACC_Y_BIAS: f32 = 0.013108;
const ACC_Z_BIAS: f32 = -0.007008;

const MAG_X_BIAS : f32 = 25968.448f32; 
const MAG_Y_BIAS : f32 = -54043.074f32;
const MAG_Z_BIAS : f32 = -16326.434f32;

#[derive(Debug, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
pub struct UartHeader { 
    frame_start  : u32,
    payload_len  : u32 
}

#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct Orientation {
    roll : f32, pitch : f32, yaw : f32,
}

bind_interrupts!(struct Irqs {
    UARTE0 => uarte::InterruptHandler<peripherals::UARTE0>;
    TWISPI1 => twim::InterruptHandler<peripherals::TWISPI1>;
    TWISPI0 => twim::InterruptHandler<peripherals::TWISPI0>;
});

static MAD_CHAN: Channel<CriticalSectionRawMutex, Orientation, 8> = Channel::new();

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_nrf::init(Default::default());

    // mpu 6050
    // uses external i2c
    // scl P0_26 -> P19
    // sda P1_00 -> P20
    let mut ext_twim_conf = twim::Config::default();
    ext_twim_conf.frequency = twim::Frequency::K100;
    static EXT_I2C_RAMBUFF: ConstStaticCell<[u8; 64]> = ConstStaticCell::new([0; 64]);
    let scl = p.P0_26;
    let sda = p.P1_00;
    let ext_i2c = Twim::new(p.TWISPI1, Irqs, sda, scl, ext_twim_conf, EXT_I2C_RAMBUFF.take());
    let mut mpu = Mpu6050::new_with_addr(ext_i2c, 0x68);

    // lsm303agr
    // uses internal i2c 
    // scl: p0_08
    // sda: p0_16
    let mut twimconf = twim::Config::default();
    twimconf.frequency = twim::Frequency::K100;
    static RAM_BUFFER: ConstStaticCell<[u8; 16]> = ConstStaticCell::new([0; 16]);
    let i2c = Twim::new(p.TWISPI0, Irqs, p.P0_16, p.P0_08, twimconf, RAM_BUFFER.take());
    let mut lsm303 = Lsm303agr::new_with_i2c(i2c);

    // external uart connection
    let mut uartconf = uarte::Config::default();
    uartconf.baudrate = uarte::Baudrate::BAUD115200;
    uartconf.parity = uarte::Parity::EXCLUDED;
    let uart = uarte::Uarte::new(p.UARTE0, p.P1_08, p.P0_06, Irqs, uartconf);
    let (ext_tx, _rx) = uart.split();

    
    // ninedof 
    if let (Ok(Ok(_)), Ok(_)) = (mpu.init(&mut Delay).with_timeout(Duration::from_secs(1)).await, lsm303.init().await) {
        if let Ok(_) = lsm303.set_mag_mode_and_odr(&mut Delay, MagMode::LowPower, MagOutputDataRate::Hz100).await {
            if let Ok(cont_lsm) = lsm303.into_mag_continuous().await {
                let _ = spawner.spawn(pubber9(mpu, cont_lsm));
                let _ = spawner.spawn(subber(ext_tx));
            }
        }
    }
}

/* 
    Task that uses the 9 dof Madgwick filter to track orientation
*/
#[embassy_executor::task]
async fn pubber9(
    mut mpu : Mpu6050<Twim<'static>>,
    mut lsm : Lsm303agr<I2cInterface<Twim<'static>>, MagContinuous>,
) {
    let mut mad_marg = MadgwickFilter::new_9dof(0.01, 0.1, 0.00);
    let mut marg_ticka = Ticker::every(Duration::from_millis(10));

    loop {
        if let (Ok(acc), Ok(gyro)) = (mpu.get_acc().await, mpu.get_gyro().await) {
            if lsm.mag_status().await.is_ok() {
                if let Ok(mag) = lsm.magnetic_field().await {
                    mad_marg.filter9(
                        acc.x - ACC_X_BIAS,   acc.y - ACC_Y_BIAS,   acc.z - ACC_Z_BIAS, 
                        gyro.x - GYRO_X_BIAS, gyro.y - GYRO_Y_BIAS, gyro.z - GYRO_Z_BIAS, 

                        // remove hard iron biases and account for axis misalignment
                        mag.x_nt() as f32 - MAG_X_BIAS,
                        mag.y_nt() as f32 - MAG_Y_BIAS,
                        -mag.z_nt() as f32 - MAG_Z_BIAS
                    );
                }
            }

            let (roll, pitch, yaw) = mad_marg.quat().eulers();
            let orientation = Orientation{roll, pitch, yaw};
            MAD_CHAN.send(orientation).await;
        }
        marg_ticka.next().await;
    }
}

/* 
    Task that uses the 6 dof Madgwick filter
*/
#[embassy_executor::task]
async fn pubber6(
    mut mpu : Mpu6050<Twim<'static>>,
) {
    let mut mad_filter = MadgwickFilter::new_6dof(0.01, 0.1);
    let mut marg_ticka = Ticker::every(Duration::from_millis(10));

    loop {
        if let (Ok(acc), Ok(gyro)) = (mpu.get_acc().await, mpu.get_gyro().await) {
            mad_filter.filter6(
                acc.x - ACC_X_BIAS,   acc.y - ACC_Y_BIAS,   acc.z - ACC_Z_BIAS, 
                gyro.x - GYRO_X_BIAS, gyro.y - GYRO_Y_BIAS, gyro.z - GYRO_Z_BIAS
            );

            let (roll, pitch, yaw) = mad_filter.quat().eulers();
            let orientation = Orientation{roll, pitch, yaw};
            MAD_CHAN.send(orientation).await;
        }
        marg_ticka.next().await;
    }
}

#[embassy_executor::task]
async fn subber(
    mut ext_uart_tx : uarte::UarteTx<'static>
) {
    let mut frame_header = UartHeader{frame_start : UART_FRAME_START, payload_len : 0};
    let mut dbuff = [0u8; 128];

    loop {
        let ori_msg = MAD_CHAN.receive().await;
        if let Ok(ser_msg) = to_slice(&ori_msg, &mut dbuff) {
            frame_header.payload_len = ser_msg.len() as u32;
            let _ = ext_uart_tx.write(frame_header.as_bytes()).await;
            let _ = ext_uart_tx.write(ser_msg).await;
        }
    }
}