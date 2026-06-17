use std::f32::consts::PI;

use wasm_bindgen::prelude::*;

const M: f32 = 1.0;
const R_E: f32 = 2.5;
// const G: f32 = 1.0;
// const K: f32 = 1000.0;
// const MU: f32 = 0.001;
const PBF_STEPS: usize = 4;
const WALL_INT_STEPS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Vec3 {
    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn mul(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

#[wasm_bindgen]
pub struct SimSolver {
    pub n: usize,
    particle_size: f32,
    kernel_r: f32,
    container_size: f32,
    mesh_size: usize,

    pos: Vec<Vec3>,
    pos_buf: Vec<Vec3>,
    vel: Vec<Vec3>,
    vel_buf: Vec<Vec3>,

    old_pos: Vec<Vec3>,
    old_pos_buf: Vec<Vec3>,

    rho: Vec<f32>,
    lambda: Vec<f32>,

    mesh_bound: Vec<usize>,

    rho0: f32,
    wall_poly6_table: Vec<f32>,
    wall_spiky_table: Vec<f32>,
    wall_visc_table: Vec<f32>,

    gravity: f32,
    p_stiff: f32,
    visocity: f32,
}

#[wasm_bindgen]
impl SimSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n: usize,
        particle_size: f32,
        container_size: f32,
        gravity: f32,
        p_stiff: f32,
        visocity: f32,
    ) -> SimSolver {
        let kernel_r = R_E * particle_size;
        let mesh_size = (container_size / kernel_r).ceil() as usize + 2;
        SimSolver {
            n,
            particle_size,
            kernel_r,
            container_size,
            mesh_size,

            pos: vec![Vec3::default(); n],
            pos_buf: vec![Vec3::default(); n],
            vel: vec![Vec3::default(); n],
            vel_buf: vec![Vec3::default(); n],
            old_pos: vec![Vec3::default(); n],
            old_pos_buf: vec![Vec3::default(); n],

            rho: vec![0.0; n],
            lambda: vec![0.0; n],

            mesh_bound: vec![0; mesh_size.pow(3) + 1],

            rho0: SimSolver::compute_rho0(),
            wall_poly6_table: SimSolver::integrate_wall_table(|x, d| {
                SimSolver::w_poly6(x) * x * (x - d)
            }),
            wall_spiky_table: SimSolver::integrate_wall_table(|x, _d| SimSolver::w_spiky(x) * x),
            wall_visc_table: SimSolver::integrate_wall_table(|x, d| {
                SimSolver::w_visc(x) * x * (x - d)
            }),

            gravity,
            p_stiff,
            visocity,
        }
    }

    pub fn pos(&self) -> *const f32 {
        self.pos.as_ptr() as *const f32
    }

    pub fn vel(&self) -> *const f32 {
        self.vel.as_ptr() as *const f32
    }

    fn w_poly6(r: f32) -> f32 {
        315.0 / (64.0 * PI * R_E.powi(9)) * (R_E.powi(2) - r.powi(2)).max(0.0).powi(3)
    }

    fn w_spiky(r: f32) -> f32 {
        45.0 / (PI * R_E.powi(6)) * (R_E - r).max(0.0).powi(2)
    }

    fn w_visc(r: f32) -> f32 {
        45.0 / (PI * R_E.powi(6)) * (R_E - r).max(0.0)
    }

    fn neighbor_iter() -> impl Iterator<Item = Vec3> {
        let size = R_E.ceil() as isize;
        (-size..=size).flat_map(move |x| {
            (-size..=size).flat_map(move |y| {
                (-size..=size).map(move |z| Vec3 {
                    x: x as f32,
                    y: y as f32,
                    z: z as f32,
                })
            })
        })
    }

    fn compute_rho0() -> f32 {
        SimSolver::neighbor_iter()
            .map(|v| SimSolver::w_poly6(v.norm()))
            .sum()
    }

    fn wall_idx_to_x(idx: usize) -> f32 {
        idx as f32 / WALL_INT_STEPS as f32
    }

    fn integrate_wall_table<F: Fn(f32, f32) -> f32>(w: F) -> Vec<f32> {
        let total_steps = (WALL_INT_STEPS as f32 * R_E).ceil() as usize;
        let dx = 1.0 / WALL_INT_STEPS as f32;
        let f = |from: usize| {
            let d = SimSolver::wall_idx_to_x(from);
            let mut prev = w(d, d);
            let mut s = 0.0;
            for i in from + 1..=total_steps {
                let x = SimSolver::wall_idx_to_x(i);
                let y = w(x, d);
                s += y + prev;
                prev = y;
            }
            s * dx * PI
        };

        (0..=total_steps).map(|i| f(i)).collect()
    }

    fn get_mesh_pos(&self, pos: &Vec3) -> (isize, isize, isize) {
        let x_idx = ((pos.x + self.container_size / 2.0) / self.kernel_r + 1.0).floor() as isize;
        let y_idx = ((pos.y + self.container_size / 2.0) / self.kernel_r + 1.0).floor() as isize;
        let z_idx = ((pos.z + self.container_size / 2.0) / self.kernel_r + 1.0).floor() as isize;
        (x_idx, y_idx, z_idx)
    }

    fn mesh_pos_to_index(&self, x_idx: isize, y_idx: isize, z_idx: isize) -> Option<usize> {
        let m = self.mesh_size;
        let x_in = (0..m as isize).contains(&x_idx);
        let y_in = (0..m as isize).contains(&y_idx);
        let z_in = (0..m as isize).contains(&z_idx);
        if x_in && y_in && z_in {
            Some(((x_idx as usize) * m + (y_idx as usize)) * m + (z_idx as usize))
        } else {
            None
        }
    }

    fn get_mesh_index(&self, pos: &Vec3) -> Option<usize> {
        let (x_idx, y_idx, z_idx) = self.get_mesh_pos(pos);
        self.mesh_pos_to_index(x_idx, y_idx, z_idx)
    }

    fn neighbor_sort(&mut self) {
        // bucket sort
        self.mesh_bound.fill(0);
        for p in self.pos.iter() {
            let idx = self.get_mesh_index(p).unwrap_or(0);
            self.mesh_bound[idx] += 1;
        }
        for i in 0..self.mesh_size.pow(3) {
            self.mesh_bound[i + 1] += self.mesh_bound[i];
        }
        // stableにするためrev
        for ((p, v), op) in self
            .pos
            .iter()
            .zip(self.vel.iter())
            .zip(self.old_pos.iter())
            .rev()
        {
            let idx = self.get_mesh_index(p).unwrap_or(0);
            let pos_idx = self.mesh_bound[idx] - 1;
            self.pos_buf[pos_idx] = *p;
            self.vel_buf[pos_idx] = *v;
            self.old_pos_buf[pos_idx] = *op;
            self.mesh_bound[idx] = pos_idx;
        }
    }

    fn local_iter(&self, pos: &Vec3) -> impl Iterator<Item = usize> {
        let (x_idx, y_idx, z_idx) = self.get_mesh_pos(pos);
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1)
                    .filter_map(move |dz| {
                        self.mesh_pos_to_index(x_idx + dx, y_idx + dy, z_idx + dz)
                            .map(|idx| {
                                let start = self.mesh_bound[idx];
                                let end = self.mesh_bound[idx + 1];
                                start..end
                            })
                    })
                    .flatten()
            })
        })
    }

    fn get_nearest_wall(&self, pos: &Vec3) -> (f32, Vec3) {
        // distとnorm(内向き)
        let half_size = self.container_size / 2.0;
        let dist_x = half_size - pos.x.abs();
        let dist_y = half_size - pos.y.abs();
        let dist_z = half_size - pos.z.abs();
        let min = dist_x.min(dist_y).min(dist_z);
        let norm = if min == dist_x {
            Vec3 {
                x: -pos.x.signum(),
                y: 0.0,
                z: 0.0,
            }
        } else if min == dist_y {
            Vec3 {
                x: 0.0,
                y: -pos.y.signum(),
                z: 0.0,
            }
        } else {
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -pos.z.signum(),
            }
        };
        (min, norm)
    }

    fn wall_rho(&self, dist: f32) -> f32 {
        if dist <= -R_E {
            return 1.0;
        }
        let idx = (dist.abs() * WALL_INT_STEPS as f32).round() as usize;
        self.wall_poly6_table
            .get(idx)
            .map_or(0.0, |v| if dist > 0.0 { *v } else { 1.0 - *v })
    }

    fn wall_press(&self, dist: f32) -> f32 {
        let idx = (dist.abs() * WALL_INT_STEPS as f32).round() as usize;
        self.wall_spiky_table.get(idx).map_or(0.0, |v: &f32| *v)
    }

    fn wall_visc(&self, dist: f32) -> f32 {
        let total = 2.0 * self.wall_visc_table[0];
        if dist <= -R_E {
            return total;
        }
        let idx = (dist.abs() * WALL_INT_STEPS as f32).round() as usize;
        self.wall_visc_table
            .get(idx)
            .map_or(0.0, |v| if dist > 0.0 { *v } else { total - *v })
    }

    fn apply_wall_constraint(pos: &mut Vec3, vel: &mut Vec3, d: f32, wall_norm: &Vec3) {
        if d < 0.0 {
            *pos = pos.add(&wall_norm.mul(-d));
            let v_n = vel.dot(wall_norm);
            if v_n < 0.0 {
                *vel = vel.sub(&wall_norm.mul(v_n));
            }
        }
    }

    fn compute_rho(&mut self) {
        for (i, p) in self.pos_buf.iter().enumerate() {
            let (d, _) = self.get_nearest_wall(p);
            let rho = self
                .local_iter(p)
                .map(|idx| {
                    let r = self.pos_buf[idx].sub(p).norm() / self.particle_size;
                    if r < R_E { SimSolver::w_poly6(r) } else { 0.0 }
                })
                .sum::<f32>()
                * M;
            self.rho[i] = rho + self.wall_rho(d / self.particle_size) * M;
        }
    }

    pub fn step_sph(&mut self, dt: f32) {
        self.neighbor_sort();

        self.compute_rho();

        for (i, p) in self.pos_buf.iter().enumerate() {
            let (d, _wall_norm) = self.get_nearest_wall(p);
            let mut a = Vec3 {
                x: 0.0,
                y: -self.gravity,
                z: 0.0,
            };
            let press_i = (self.p_stiff * (self.rho[i] - self.rho0)).max(0.0);
            let qd = d / self.particle_size;

            // a = a.add(&wall_norm.mul(
            //     M * (press_i / self.rho[i].powi(2) + press_i / self.rho0.powi(2))
            //         * self.wall_press(qd)
            //         / self.particle_size,
            // ));
            a = a.add(&self.vel_buf[i].mul(
                -M * self.visocity / self.rho0 * self.wall_visc(qd) / self.particle_size.powi(2),
            ));

            for (j, q) in self.local_iter(p).map(|idx| (idx, self.pos_buf[idx])) {
                if i == j {
                    continue;
                }
                let press_j = (self.p_stiff * (self.rho[j] - self.rho0)).max(0.0);
                let r = q.sub(p);
                let r_len = r.norm();
                if r_len >= R_E * self.particle_size || r_len < 1e-6 {
                    continue;
                }
                let q_len = r_len / self.particle_size;
                let rv = self.vel_buf[j].sub(&self.vel_buf[i]);
                a = a.sub(&r.mul(
                    M * (press_i / self.rho[i].powi(2) + press_j / self.rho[j].powi(2))
                        * SimSolver::w_spiky(q_len)
                        / (r_len * self.particle_size),
                ));
                a = a.add(&rv.mul(
                    M * self.visocity / self.rho[j] * SimSolver::w_visc(q_len)
                        / self.particle_size.powi(2),
                ));
            }

            self.vel[i] = self.vel_buf[i].add(&a.mul(dt));
            self.pos[i] = self.pos_buf[i].add(&self.vel[i].mul(dt));

            let (d, wall_norm) = self.get_nearest_wall(&self.pos[i]);
            SimSolver::apply_wall_constraint(&mut self.pos[i], &mut self.vel[i], d, &wall_norm);
        }
    }

    pub fn step_pbf(&mut self, dt: f32) {
        self.old_pos.copy_from_slice(&self.pos);

        let s = M / self.rho0;

        for i in 0..self.n {
            self.vel[i] = self.vel[i].add(&Vec3 {
                x: 0.0,
                y: -self.gravity * dt,
                z: 0.0,
            });
            self.pos[i] = self.pos[i].add(&self.vel[i].mul(dt));

            let (d, wall_norm) = self.get_nearest_wall(&self.pos[i]);
            SimSolver::apply_wall_constraint(&mut self.pos[i], &mut self.vel[i], d, &wall_norm);
        }

        for _ in 0..PBF_STEPS {
            self.neighbor_sort();
            self.compute_rho();

            for (i, p) in self.pos_buf.iter().enumerate() {
                let c_i = (self.rho[i] / self.rho0 - 1.0).max(0.0);

                let mut wij_sum = Vec3::default();
                let mut wij_norm_sum = 0.0;
                for j in self.local_iter(p) {
                    let q = self.pos_buf[j];
                    let r = q.sub(p);
                    let r_len = r.norm();
                    if r_len >= R_E * self.particle_size || r_len < 1e-6 {
                        continue;
                    }
                    let wij = r.mul(
                        SimSolver::w_spiky(r_len / self.particle_size)
                            / (r_len * self.particle_size),
                    );
                    wij_sum = wij_sum.add(&wij);
                    wij_norm_sum += wij.norm().powi(2);
                }
                let (d, wall_norm) = self.get_nearest_wall(p);
                let wi_wall =
                    wall_norm.mul(-self.wall_press(d / self.particle_size) / self.particle_size);
                let self_w = wij_sum.add(&wi_wall);

                self.lambda[i] = -c_i / (s * (self_w.dot(&self_w) + wij_norm_sum) + 1e-6);
            }

            for (i, p) in self.pos_buf.iter().enumerate() {
                let (d, wall_norm) = self.get_nearest_wall(p);
                let wi_wall =
                    wall_norm.mul(-self.wall_press(d / self.particle_size) / self.particle_size);

                let mut delta = wi_wall.mul(self.lambda[i]);
                for j in self.local_iter(p) {
                    let q = self.pos_buf[j];
                    let r = q.sub(p);
                    let r_len = r.norm();
                    if r_len >= R_E * self.particle_size || r_len < 1e-6 {
                        continue;
                    }
                    let wij = r.mul(
                        SimSolver::w_spiky(r_len / self.particle_size)
                            / (r_len * self.particle_size),
                    );
                    delta = delta.add(&wij.mul(self.lambda[i] + self.lambda[j]));
                }
                self.pos[i] = self.pos_buf[i].add(&delta);
                self.vel[i] = self.vel_buf[i];

                let (d, wall_norm) = self.get_nearest_wall(&self.pos[i]);
                SimSolver::apply_wall_constraint(&mut self.pos[i], &mut self.vel[i], d, &wall_norm);
            }

            self.old_pos.copy_from_slice(&self.old_pos_buf);
        }

        self.neighbor_sort();
        self.compute_rho();
        self.pos.copy_from_slice(&self.pos_buf);

        for i in 0..self.n {
            self.vel_buf[i] = self.pos_buf[i].sub(&self.old_pos_buf[i]).mul(1.0 / dt);
        }

        for i in 0..self.n {
            let (d, wall_norm) = self.get_nearest_wall(&self.pos[i]);
            let mut vis = self.vel_buf[i]
                .mul(-s * self.wall_visc(d / self.particle_size) / self.particle_size.powi(2));
            for j in self.local_iter(&self.pos_buf[i]) {
                let q = self.pos_buf[j];
                let r = q.sub(&self.pos_buf[i]);
                let r_len = r.norm();
                if r_len >= R_E * self.particle_size || r_len < 1e-6 {
                    continue;
                }
                let rv = self.vel_buf[j].sub(&self.vel_buf[i]);
                vis = vis.add(&rv.mul(
                    M / self.rho[j] * SimSolver::w_visc(r_len / self.particle_size)
                        / self.particle_size.powi(2),
                ));
            }
            self.vel[i] = self.vel_buf[i].add(&vis.mul(dt * self.visocity));

            SimSolver::apply_wall_constraint(&mut self.pos[i], &mut self.vel[i], d, &wall_norm);
        }
    }
}
