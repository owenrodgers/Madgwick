use crate::quaternion::Quaternion;

/*  
    Adapted from Blake Johnsons implementation in C
    https://github.com/bjohnsonfl/Madgwick_Filter

    Algortithm spec:
    https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
*/

pub const BETA_6DOF : f32 = 0.033;
pub const BETA_9DOF : f32 = 0.041;

pub trait Madgwick {
    fn filter(&mut self, ax : f32, ay : f32, az : f32, gx : f32, gy : f32, gz : f32) -> ();
    fn quat(&self) -> Quaternion;
}

pub struct MadgwickFilter {
    delta_t : f32,
    beta : f32,

    estimate : Quaternion,
}

impl Madgwick for MadgwickFilter {
    fn filter(&mut self, ax : f32, ay : f32, az : f32, gx : f32, gy : f32, gz : f32) -> () {
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
    pub fn new(delta_t : f32, beta : f32) -> MadgwickFilter {
        MadgwickFilter{delta_t, beta,
            estimate : Quaternion::new(1.0, 0.0, 0.0, 0.0)}
    }
}