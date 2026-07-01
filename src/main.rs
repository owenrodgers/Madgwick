#![no_std]
#![no_main]

mod quaternion;
mod madgwick;

use defmt_rtt as _;
use mpu6050_async::Mpu6050;
use panic_probe as _;
use postcard::to_slice;
use static_cell::ConstStaticCell;
use embassy_executor::Spawner;
use embassy_time::{Delay, Duration, WithTimeout};
use embassy_nrf::{bind_interrupts, peripherals, twim::{self, Twim}, uarte::{self}};
use serde::{Deserialize, Serialize};
use zerocopy::{IntoBytes, Immutable, KnownLayout};
use madgwick::MadgwickFilter;

use crate::madgwick::Madgwick;

const UART_FRAME_START : u32 = 0x77_77_77_77;

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
});

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

    // external uart connection
    let mut uartconf = uarte::Config::default();
    uartconf.baudrate = uarte::Baudrate::BAUD115200;
    uartconf.parity = uarte::Parity::EXCLUDED;
    let uart = uarte::Uarte::new(p.UARTE0, p.P1_08, p.P0_06, Irqs, uartconf);
    let (tx, _rx) = uart.split();

    if let Ok(Ok(_)) = mpu.init(&mut Delay).with_timeout(Duration::from_secs(1)).await {
        let _ = spawner.spawn(imu(mpu, tx));
    }
}

#[embassy_executor::task]
async fn imu(
    mut mpu : Mpu6050<Twim<'static>>,
    mut uart : uarte::UarteTx<'static>
) {
    let mut mad_filter = MadgwickFilter::new(0.01, madgwick::BETA_6DOF);
    let mut frame_header = UartHeader{frame_start : UART_FRAME_START, payload_len : 0};
    let mut dbuff = [0u8; 128];

    loop {
        if let (Ok(acc), Ok(gyro)) = (mpu.get_acc().await, mpu.get_gyro().await) {
            mad_filter.filter(acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z);
            let (roll, pitch, yaw) = mad_filter.quat().eulers();
            let orientation = Orientation{roll, pitch, yaw};

            if let Ok(ser_msg) = to_slice(&orientation, &mut dbuff) {
                frame_header.payload_len = ser_msg.len() as u32;
                let _ = uart.write(frame_header.as_bytes()).await;
                let _ = uart.write(ser_msg).await;
            }
        }
    }
}