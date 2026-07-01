use core::ops;
use num_traits::Float;

const PI : f32 = 3.14159265359;

/* 
    Quaternions have 4 components, 1 real and 3 imaginary.
    a + bi + cj + dk
*/
#[derive(Debug, Default, Copy, Clone)]
pub struct Quaternion {
    pub a : f32, 
    pub b : f32, 
    pub c : f32, 
    pub d : f32
}

impl Quaternion {
    pub fn new(a : f32, b : f32, c : f32, d : f32) -> Quaternion {
        Quaternion{a, b, c, d}
    }

    pub fn conjugate(&self) -> Quaternion {
        Quaternion::new(self.a, -self.b, -self.c, -self.d)
    }

    pub fn norm(&self) -> f32 {
        (self.a * self.a + 
            self.b * self.b + 
            self.c * self.c + 
            self.d * self.d
        ).sqrt()
    }

    pub fn normalize(&self) -> Quaternion {
        let norm : f32 = self.norm();
        match norm {
            0.0 => Quaternion::new(self.a, self.b, self.c, self.d),
            _ => Quaternion::new(self.a / norm, self.b / norm, self.c / norm, self.d / norm)
        }
    }

    pub fn eulers(&self) -> (f32, f32, f32) {
        let yaw : f32 = (2.0 * (self.b * self.c - self.a * self.d))
            .atan2(1.0 - 2.0 * (self.a * self.a + self.b * self.b));

        let sin_pitch = 2.0 * (self.a * self.c - self.d * self.b);
        let pitch = sin_pitch.clamp(-1.0, 1.0).asin();

        let roll = (2.0 * (self.a * self.b + self.c * self.d))
            .atan2(1.0 - 2.0 * (self.b * self.b + self.c * self.c));

        (roll * (180.0 / PI), (pitch * (180.0 / PI)), yaw * (180.0 / PI))
    }
}

/* 
    Component wise addition and subtraction
*/
impl ops::Add for Quaternion {
    type Output = Quaternion;

    fn add(self, rhs: Self) -> Self::Output {
        Quaternion::new(
            self.a + rhs.a, 
            self.b + rhs.b, 
            self.c + rhs.c, 
            self.d + rhs.d
        )
    }
}

impl ops::Sub for Quaternion {
    type Output = Quaternion;

    fn sub(self, rhs: Self) -> Self::Output {
        Quaternion::new(
            self.a - rhs.a, 
            self.b - rhs.b, 
            self.c - rhs.c, 
            self.d - rhs.d
        )
    }
}

/* 
    Scalar multiply
*/
impl ops::Mul<Quaternion> for f32 {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Self::Output {
        Quaternion::new(
            self * rhs.a, 
            self * rhs.b, 
            self * rhs.c, 
            self * rhs.d
        )
    }
}

impl ops::MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, rhs: f32) {
        self.a *= rhs;
        self.b *= rhs;
        self.c *= rhs;
        self.d *= rhs;
    }
}

/*
    Product
*/
impl ops::Mul for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Self::Output {
        Quaternion::new(
            (self.a * rhs.a) - (self.b * rhs.b) - (self.c * rhs.c) - (self.d * rhs.d), 
            (self.a * rhs.b) + (self.b * rhs.a) + (self.c * rhs.d) - (self.d * rhs.c), // i
            (self.a * rhs.c) - (self.b * rhs.d) + (self.c * rhs.a) + (self.d * rhs.b), // j
            (self.a * rhs.d) + (self.b * rhs.c) - (self.c * rhs.b) + (self.d * rhs.a)  // k
        )
    }
}
