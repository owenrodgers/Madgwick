use crate::quaternion::Quaternion;
use num_traits::Float;

/*  
    Algortithm spec:
    https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
*/

pub trait Madgwick {
    fn filter6(&mut self, ax : f32, ay : f32, az : f32, gx : f32, gy : f32, gz : f32) -> ();
    fn filter9(&mut self, 
        ax : f32, ay : f32, az : f32, 
        gx : f32, gy : f32, gz : f32,
        mx : f32, my : f32, mz : f32
    ) -> ();
    fn quat(&self) -> Quaternion;
}

pub struct MadgwickFilter {
    delta_t : f32,
    beta : f32,
    zeta : f32,
    gyro_error : Quaternion,

    estimate : Quaternion,
}

impl Madgwick for MadgwickFilter {
    fn filter9(&mut self, 
        ax : f32, ay : f32, az : f32, 
        gx : f32, gy : f32, gz : f32,
        mx : f32, my : f32, mz : f32
    ) -> ()
    {
        let q_est_t_1 = self.estimate.clone();

        // normed measurements
        let a_t = Quaternion::new(0.0, ax, ay, az).normalize();
        let m_t = Quaternion::new(0.0, mx, my, mz).normalize();
        let w_t = Quaternion::new(0.0, gx, gy, gz);

        // compensate for magnetic drift in m_t (group 1)
        // (45) and (46)
        let h_t = q_est_t_1 * m_t * q_est_t_1.conjugate(); // (eq 45)
        let b_t = Quaternion::new(0.0, (h_t.b * h_t.b + h_t.c * h_t.c).sqrt(), 0.0, h_t.d); // (eq 46)

        // build J + f
        // there are two pieces to J_gb and f_gb, the g part and b part

        // b (magnetic) part
        // f_b (equation 29)
        let f_b : [f32 ; 3] = [
            2.0 * b_t.b * (0.5 - q_est_t_1.c * q_est_t_1.c - q_est_t_1.d * q_est_t_1.d) + 2.0 * b_t.d * (q_est_t_1.b * q_est_t_1.d - q_est_t_1.a * q_est_t_1.c) - m_t.b,
            2.0 * b_t.b * (q_est_t_1.b * q_est_t_1.c - q_est_t_1.a * q_est_t_1.d) + 2.0 * b_t.d * (q_est_t_1.a * q_est_t_1.b + q_est_t_1.c * q_est_t_1.d) - m_t.c,
            2.0 * b_t.b * (q_est_t_1.a * q_est_t_1.c + q_est_t_1.b * q_est_t_1.d) + 2.0 * b_t.d * (0.5 - q_est_t_1.b * q_est_t_1.b - q_est_t_1.c * q_est_t_1.c) - m_t.d
        ];

        let mut jacobian_b = [[0f32; 4]; 3];
        jacobian_b[0][0] = -2.0 * b_t.d * q_est_t_1.c;
        jacobian_b[0][1] = 2.0 * b_t.d * q_est_t_1.d;
        jacobian_b[0][2] = -4.0 * b_t.b * q_est_t_1.c - 2.0 * b_t.d * q_est_t_1.a;
        jacobian_b[0][3] = -4.0 * b_t.b * q_est_t_1.d + 2.0 * b_t.d * q_est_t_1.b;

        jacobian_b[1][0] = -2.0 * b_t.b * q_est_t_1.d + 2.0 * b_t.d * q_est_t_1.b;
        jacobian_b[1][1] = 2.0 * b_t.b * q_est_t_1.c + 2.0 * b_t.d * q_est_t_1.a;
        jacobian_b[1][2] = 2.0 * b_t.b * q_est_t_1.b + 2.0 * b_t.d * q_est_t_1.d;
        jacobian_b[1][3] = -2.0 * b_t.b * q_est_t_1.a + 2.0 * b_t.d * q_est_t_1.c;

        jacobian_b[2][0] = 2.0 * b_t.b * q_est_t_1.c;
        jacobian_b[2][1] = 2.0 * b_t.b * q_est_t_1.d - 4.0 * b_t.d * q_est_t_1.b;
        jacobian_b[2][2] = 2.0 * b_t.b * q_est_t_1.a - 4.0 * b_t.d * q_est_t_1.c;
        jacobian_b[2][3] = 2.0 * b_t.b * q_est_t_1.b;

        let mag_grad = Quaternion::new(
            jacobian_b[0][0] * f_b[0] + jacobian_b[1][0] * f_b[1] + jacobian_b[2][0] * f_b[2], 
            jacobian_b[0][1] * f_b[0] + jacobian_b[1][1] * f_b[1] + jacobian_b[2][1] * f_b[2], 
            jacobian_b[0][2] * f_b[0] + jacobian_b[1][2] * f_b[1] + jacobian_b[2][2] * f_b[2], 
            jacobian_b[0][3] * f_b[0] + jacobian_b[1][3] * f_b[1] + jacobian_b[2][3] * f_b[2]
        );

        // f_g (equation 25)
        let f_g : [f32; 3] = [
            2.0 * (q_est_t_1.b * q_est_t_1.d - q_est_t_1.a * q_est_t_1.c) - a_t.b,
            2.0 * (q_est_t_1.a * q_est_t_1.b + q_est_t_1.c * q_est_t_1.d) - a_t.c,
            2.0 * (0.5 - q_est_t_1.b * q_est_t_1.b - q_est_t_1.c * q_est_t_1.c) - a_t.d,
        ];

        let mut jacobian_g = [[0f32 ; 4] ; 3];
        jacobian_g[0][0] = -2.0 * q_est_t_1.c;
        jacobian_g[0][1] = 2.0 * q_est_t_1.d;
        jacobian_g[0][2] = -2.0 * q_est_t_1.a;
        jacobian_g[0][3] = 2.0 * q_est_t_1.b;

        jacobian_g[1][0] = 2.0 * q_est_t_1.b;
        jacobian_g[1][1] = 2.0 * q_est_t_1.a;
        jacobian_g[1][2] = 2.0 * q_est_t_1.d;
        jacobian_g[1][3] = 2.0 * q_est_t_1.c;

        jacobian_g[2][0] = 0.0;
        jacobian_g[2][1] = -4.0 * q_est_t_1.b;
        jacobian_g[2][2] = -4.0 * q_est_t_1.c;
        jacobian_g[2][3] = 0.0;

        let acc_grad = Quaternion::new(
            jacobian_g[0][0] * f_g[0] + jacobian_g[1][0] * f_g[1] + jacobian_g[2][0] * f_g[2], 
            jacobian_g[0][1] * f_g[0] + jacobian_g[1][1] * f_g[1] + jacobian_g[2][1] * f_g[2], 
            jacobian_g[0][2] * f_g[0] + jacobian_g[1][2] * f_g[1] + jacobian_g[2][2] * f_g[2], 
            jacobian_g[0][3] * f_g[0] + jacobian_g[1][3] * f_g[1] + jacobian_g[2][3] * f_g[2]
        );

        let grad_f = (mag_grad + acc_grad).normalize();

        // (44) says grad f === S E q hat dot epsilon t
        // omega_error = omega_error + zeta * ((conjugate prev. estimate) * (S E q hat dot epsilon t)) * delta_t
        // omega_compensated = omega_measured - omega_error
        // (47) (48) and (49)
        self.gyro_error = self.gyro_error + self.zeta * ((2.0 * q_est_t_1.conjugate() * grad_f) * self.delta_t);
        let mut omega_compensated = w_t - self.gyro_error;
        omega_compensated.a = 0.0;

        // fuse everything together
        self.estimate = (q_est_t_1 +
            (((0.5 * q_est_t_1 * omega_compensated) - (self.beta * grad_f)) * self.delta_t)
        ).normalize();
    }

    fn filter6(&mut self, ax : f32, ay : f32, az : f32, gx : f32, gy : f32, gz : f32) -> () {
        let prev_estimate = self.estimate.clone();
        let q_w = prev_estimate * (0.5 * Quaternion::new(0.0, gx, gy, gz));
        let normed_acc = Quaternion::new(0.0, ax, ay, az).normalize();

        let mut f_g = [0f32 ; 3];
        f_g[0] = 2.0 * (prev_estimate.b * prev_estimate.d - prev_estimate.a * prev_estimate.c) - normed_acc.b;
        f_g[1] = 2.0 * (prev_estimate.a * prev_estimate.b + prev_estimate.c * prev_estimate.d) - normed_acc.c;
        f_g[2] = 2.0 * (0.5 - prev_estimate.b * prev_estimate.b - prev_estimate.c * prev_estimate.c) - normed_acc.d;

        let mut jacobian_g = [[0f32 ; 4] ; 3];
        jacobian_g[0][0] = -2.0 * prev_estimate.c;
        jacobian_g[0][1] = 2.0 * prev_estimate.d;
        jacobian_g[0][2] = -2.0 * prev_estimate.a;
        jacobian_g[0][3] = 2.0 * prev_estimate.b;

        jacobian_g[1][0] = 2.0 * prev_estimate.b;
        jacobian_g[1][1] = 2.0 * prev_estimate.a;
        jacobian_g[1][2] = 2.0 * prev_estimate.d;
        jacobian_g[1][3] = 2.0 * prev_estimate.c;

        jacobian_g[2][0] = 0.0;
        jacobian_g[2][1] = -4.0 * prev_estimate.b;
        jacobian_g[2][2] = -4.0 * prev_estimate.c;
        jacobian_g[2][3] = 0.0;

        let gradient = Quaternion::new(
            jacobian_g[0][0] * f_g[0] + jacobian_g[1][0] * f_g[1] + jacobian_g[2][0] * f_g[2], 
            jacobian_g[0][1] * f_g[0] + jacobian_g[1][1] * f_g[1] + jacobian_g[2][1] * f_g[2], 
            jacobian_g[0][2] * f_g[0] + jacobian_g[1][2] * f_g[1] + jacobian_g[2][2] * f_g[2], 
            jacobian_g[0][3] * f_g[0] + jacobian_g[1][3] * f_g[1] + jacobian_g[2][3] * f_g[2]
        ).normalize();
        
        let q_est_dot = q_w - (self.beta * gradient);
        self.estimate = (prev_estimate + (self.delta_t * q_est_dot)).normalize();
    }

    fn quat(&self) -> Quaternion {
        self.estimate
    }
}

impl MadgwickFilter {
    pub fn new_6dof(delta_t : f32, beta : f32) -> MadgwickFilter {
        MadgwickFilter{delta_t, beta,
            estimate : Quaternion::new(1.0, 0.0, 0.0, 0.0),
            gyro_error : Quaternion { a: 0.0, b: 0.0, c: 0.0, d: 0.0},
            zeta : 0.0,}
    }

    pub fn new_9dof(delta_t : f32, beta : f32, zeta : f32) -> MadgwickFilter {
        MadgwickFilter{delta_t, beta, zeta,
            gyro_error : Quaternion { a: 0.0, b: 0.0, c: 0.0, d: 0.0},
            estimate : Quaternion::new(1.0, 0.0, 0.0, 0.0)}
    }
}