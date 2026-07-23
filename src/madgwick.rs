#![allow(dead_code)]
use nalgebra::{Matrix3x4, Quaternion, RowVector4, UnitQuaternion, Vector3};
use num_traits::Float;

/*  
    Algortithm spec:
    https://x-io.co.uk/downloads/madgwick_internal_report.pdf
*/
pub trait Madgwick {
    fn filter6(&mut self, 
        acc : Vector3<f32>,
        gyro : Vector3<f32>
    );
    fn filter9(&mut self, 
        acc : Vector3<f32>,
        gyro : Vector3<f32>,
        mag : Vector3<f32>
    );
    fn quat(&self) -> UnitQuaternion<f32>;
}

pub struct MadgwickFilter {
    delta_t : f32,
    beta : f32,
    zeta : f32,
    gyro_error : Quaternion<f32>,
    estimate : UnitQuaternion<f32>,
}

impl Madgwick for MadgwickFilter {
    fn filter9(&mut self, 
        acc : Vector3<f32>,
        gyro : Vector3<f32>,
        mag : Vector3<f32>
    ) {
        let q_est_t_1 = self.estimate.into_inner();

        // normed measurements
        let m_t = Quaternion::new(0.0, mag.x, mag.y, mag.z).normalize();
        let w_t = Quaternion::new(0.0, gyro.x, gyro.y, gyro.z);

        // compensate for magnetic drift in m_t (group 1)
        // (45) and (46)
        let h_t = q_est_t_1 * m_t * q_est_t_1.conjugate(); // (eq 45)
        let b_t = Quaternion::new(
            0.0,
            (h_t.i * h_t.i + h_t.j * h_t.j).sqrt(),
            0.0,
            h_t.k,
        ); // (eq 46)

        // build J + f
        // there are two pieces to J_gb and f_gb, the g part and b part

        // b (magnetic) part
        // f_b (equation 29)
        let f_b = Vector3::new(
            2.0 * b_t.i * (0.5 - q_est_t_1.j * q_est_t_1.j - q_est_t_1.k * q_est_t_1.k)
                + 2.0 * b_t.k * (q_est_t_1.i * q_est_t_1.k - q_est_t_1.w * q_est_t_1.j),
            2.0 * b_t.i * (q_est_t_1.i * q_est_t_1.j - q_est_t_1.w * q_est_t_1.k)
                + 2.0 * b_t.k * (q_est_t_1.w * q_est_t_1.i + q_est_t_1.j * q_est_t_1.k),
            2.0 * b_t.i * (q_est_t_1.w * q_est_t_1.j + q_est_t_1.i * q_est_t_1.k)
                + 2.0 * b_t.k * (0.5 - q_est_t_1.i * q_est_t_1.i - q_est_t_1.j * q_est_t_1.j)
        ) - mag.normalize();
 
        let jacobian_b = Matrix3x4::from_rows(&[
            RowVector4::new(-2.0 * b_t.k * q_est_t_1.j, 2.0 * b_t.k * q_est_t_1.k, -4.0 * b_t.i * q_est_t_1.j - 2.0 * b_t.k * q_est_t_1.w, -4.0 * b_t.i * q_est_t_1.k + 2.0 * b_t.k * q_est_t_1.i),
            RowVector4::new(-2.0 * b_t.i * q_est_t_1.k + 2.0 * b_t.k * q_est_t_1.i, 2.0 * b_t.i * q_est_t_1.j + 2.0 * b_t.k * q_est_t_1.w, 2.0 * b_t.i * q_est_t_1.i + 2.0 * b_t.k * q_est_t_1.k, -2.0 * b_t.i * q_est_t_1.w + 2.0 * b_t.k * q_est_t_1.j),
            RowVector4::new(2.0 * b_t.i * q_est_t_1.j, 2.0 * b_t.i * q_est_t_1.k - 4.0 * b_t.k * q_est_t_1.i, 2.0 * b_t.i * q_est_t_1.w - 4.0 * b_t.k * q_est_t_1.j, 2.0 * b_t.i * q_est_t_1.i)
        ]);
 
        let p = jacobian_b.transpose() * f_b;
        let mag_grad = Quaternion::new(p[0], p[1], p[2], p[3]);
 
        // f_g (equation 25)
        let f_g = Vector3::new(
            2.0 * (q_est_t_1.i * q_est_t_1.k - q_est_t_1.w * q_est_t_1.j), 
            2.0 * (q_est_t_1.w * q_est_t_1.i + q_est_t_1.j * q_est_t_1.k), 
            2.0 * (0.5 - q_est_t_1.i * q_est_t_1.i - q_est_t_1.j * q_est_t_1.j)
        ) - acc.normalize();
 
        let jacobian_g = Matrix3x4::from_rows(&[
            RowVector4::new(-2.0 * q_est_t_1.j, 2.0 * q_est_t_1.k, -2.0 * q_est_t_1.w, 2.0 * q_est_t_1.i),
            RowVector4::new(2.0 * q_est_t_1.i, 2.0 * q_est_t_1.w, 2.0 * q_est_t_1.k, 2.0 * q_est_t_1.j),
            RowVector4::new(0.0, -4.0 * q_est_t_1.i, -4.0 * q_est_t_1.j, 0.0)
        ]);
 
        let p = jacobian_g.transpose() * f_g;
        let acc_grad = Quaternion::new(p[0], p[1], p[2], p[3]);
 
        let grad_f = (mag_grad + acc_grad).normalize();

        // (44) says grad f === S E q hat dot epsilon t
        // omega_error = omega_error + zeta * ((conjugate prev. estimate) * (S E q hat dot epsilon t)) * delta_t
        // omega_compensated = omega_measured - omega_error
        // (47) (48) and (49)
        self.gyro_error = self.gyro_error + self.zeta * ((2.0 * q_est_t_1.conjugate() * grad_f) * self.delta_t);
        let mut omega_compensated = w_t - self.gyro_error;
        omega_compensated.w = 0.0;

        // fuse everything together
        let q_est_t = q_est_t_1 +
            (((0.5 * q_est_t_1 * omega_compensated) - (self.beta * grad_f)) * self.delta_t);
        self.estimate = UnitQuaternion::from_quaternion(q_est_t);
    }

    fn filter6(&mut self, 
        acc : Vector3<f32>,
        gyro : Vector3<f32>
    ){
        let prev_estimate: Quaternion<f32> = self.estimate.into_inner();
        let q_w = prev_estimate * (0.5 * Quaternion::new(0.0, gyro.x, gyro.y, gyro.z));

        let f_g = Vector3::new(
            2.0 * (prev_estimate.i * prev_estimate.k - prev_estimate.w * prev_estimate.j), 
            2.0 * (prev_estimate.w * prev_estimate.i + prev_estimate.j * prev_estimate.k), 
            2.0 * (0.5 - prev_estimate.i * prev_estimate.i - prev_estimate.j * prev_estimate.j)
        ) - acc.normalize();
 
        let jacobian_g = Matrix3x4::from_rows(&[
            RowVector4::new(-2.0 * prev_estimate.j, 2.0 * prev_estimate.k, -2.0 * prev_estimate.w, 2.0 * prev_estimate.i),
            RowVector4::new(2.0 * prev_estimate.i, 2.0 * prev_estimate.w, 2.0 * prev_estimate.k, 2.0 * prev_estimate.j),
            RowVector4::new(0.0, -4.0 * prev_estimate.i, -4.0 * prev_estimate.j, 0.0)
        ]);
        let gradient = Quaternion::from(jacobian_g.transpose() * f_g).normalize();
 
        let q_est_dot = q_w - (self.beta * gradient);
        let updated = prev_estimate + (self.delta_t * q_est_dot);
        self.estimate = UnitQuaternion::from_quaternion(updated);
    }

    fn quat(&self) -> UnitQuaternion<f32> {
        self.estimate
    }
}

impl MadgwickFilter {
    pub fn new_6dof(delta_t : f32, beta : f32) -> MadgwickFilter {
        MadgwickFilter{delta_t, beta,
            estimate : UnitQuaternion::new_normalize(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            gyro_error : Quaternion::new(0.0, 0.0, 0.0, 0.0),
            zeta : 0.0,}
    }

    pub fn new_9dof(delta_t : f32, beta : f32, zeta : f32) -> MadgwickFilter {
        MadgwickFilter{delta_t, beta, zeta,
            estimate : UnitQuaternion::new_normalize(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            gyro_error : Quaternion::new(0.0, 0.0, 0.0, 0.0)}
    }
}