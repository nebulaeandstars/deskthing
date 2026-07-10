use crate::frame::Layout;
use crate::grid::Grid;
use crate::traits::*;
use crate::Frame;

use macroquad::prelude::*;
use miniquad::graphics::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};

const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 100.0;

const SMOOTHING_RADIUS: f32 = 50.;
const GAS_CONSTANT: f32 = 0.05;
const REST_DENSITY: f32 = 0.;
const GRAVITY_FACTOR: f32 = 100.;

const PARTICLE_MASS: f32 = 0.1;

const RESTITUTION_COEFFICIENT: f32 = 0.8;

const PARTICLE_DRAW_SIZE: f32 = 5.;

const VERTEX_SHADER: &str = r#"
#version 100

precision lowp float;

attribute vec3 position;
attribute vec2 texcoord;

varying vec2 uv;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
}
"#;

const FRAGMENT_SHADER: &str = r#"
#version 100
precision lowp float; precision mediump int;

varying lowp vec2 uv;

uniform sampler2D Texture;

float sample_density(vec2 offset) {
    return texture2D(Texture, uv + offset).r;
}

void main() {
    float blur_radius = 5.0;
    float px = blur_radius / 512.0;
    vec3 color = vec3(0.1,0.5,1.0);

    // Calculate the now-blurred density for the pixel.
    float density = 0.0;
    density += sample_density(vec2(-px, 0.0));
    density += sample_density(vec2(px, 0.0));
    density += sample_density(vec2(0.0, -px));
    density += sample_density(vec2(0.0, px));
    density += texture2D(Texture, uv).r * 4.0;
    density /= 8.0;
    density *= 1.0;

    // Make it blobby
    float alpha = smoothstep(0.2, 0.8, density);

    // Estimate pixel "direction" for normal mapping
    float dx =
        sample_density(vec2(px,0.0)) -
        sample_density(vec2(-px,0.0));
    float dy =
        sample_density(vec2(0.0,px)) -
        sample_density(vec2(0.0,-px));

    // Add "reflections"
    // vec2 normal = normalize(vec2(dx,dy));
    // float light =
    //     dot(normal, vec2(1.0,1.0));
    // color = color * light;

    gl_FragColor = vec4(
        color[0],
        color[1],
        color[2],
        alpha
    );
}
"#;

#[derive(Clone, Debug)]
pub struct FluidParticle {
    index: usize,
    pos: Vec2,
    vel: Vec2,
}

impl FluidParticle {
    pub fn new(index: usize, x: f32, y: f32) -> Self {
        Self {
            index,
            pos: Vec2::new(x, y),
            vel: Vec2::new(0., 0.),
        }
    }

    pub fn clamp_speed(&mut self) {
        if self.vel.length() < MIN_SPEED {
            self.vel = self.vel.normalize() * MIN_SPEED
        } else if self.vel.length() > MAX_SPEED {
            self.vel = self.vel.normalize() * MAX_SPEED
        }
    }

    pub fn bounce(&mut self, frame: &Frame) {
        const MIN_DISTANCE_FROM_EDGE: f32 = 1. + PARTICLE_DRAW_SIZE / 2.;

        if self.pos.x < MIN_DISTANCE_FROM_EDGE {
            self.pos.x = MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.x > frame.width() - MIN_DISTANCE_FROM_EDGE {
            self.pos.x = frame.width() - MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        }

        if self.pos.y < MIN_DISTANCE_FROM_EDGE {
            self.pos.y = MIN_DISTANCE_FROM_EDGE;
            self.vel.y *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.y > frame.height() - MIN_DISTANCE_FROM_EDGE {
            self.pos.y = frame.height() - MIN_DISTANCE_FROM_EDGE;
            self.vel.y *= -RESTITUTION_COEFFICIENT;
        }
    }

    pub fn update<'a>(&mut self, deltatime: Duration, force: Vec2, frame: &Frame) {
        // Update velocity
        if force.is_finite() {
            self.vel += force * deltatime.as_secs_f32() / PARTICLE_MASS;
            self.clamp_speed();
        }

        // Update position
        self.pos += self.vel * deltatime.as_secs_f32();
        self.bounce(frame);
    }
}

pub struct SimpleFluidSim {
    layout: Layout,
    particles: Vec<FluidParticle>,
    densities: Vec<f32>,
    pressures: Vec<f32>,
    forces: Vec<Vec2>,
    chunks: Grid<Vec<usize>>,
    last_update: Instant,
    draw_material: Material,
}

impl SimpleFluidSim {
    pub fn new(mut layout: Layout, particles: Vec<FluidParticle>) -> Self {
        layout.refresh();

        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (layout.frame.width() / ideal_chunk_size).floor() as usize;
        let rows = (layout.frame.height() / ideal_chunk_size).floor() as usize;

        // Render material
        let draw_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: FRAGMENT_SHADER,
            },
            MaterialParams {
                uniforms: vec![],
                pipeline_params: PipelineParams {
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::Value(BlendValue::SourceAlpha),
                        BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

        // Helper states
        let num_particles = particles.len();
        let mut densities = Vec::with_capacity(num_particles);
        let mut pressures = Vec::with_capacity(num_particles);
        let mut forces = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            densities.push(0.);
            pressures.push(0.);
            forces.push(Vec2::ZERO);
        }

        Self {
            layout,
            particles,
            densities,
            pressures,
            forces,
            chunks: Grid::with_defaults(columns, rows),
            last_update: Instant::now(),
            draw_material,
        }
    }

    pub fn init(mut layout: Layout, num_particles: usize) -> Self {
        layout.refresh();

        let mut particles = Vec::new();

        for i in 0..num_particles {
            let x = rand::gen_range(100., layout.frame.width() - 100.);
            let y = rand::gen_range(100., layout.frame.height() - 100.);
            particles.push(FluidParticle::new(i, x, y));
        }

        Self::new(layout, particles)
    }

    fn update_chunks(&mut self) {
        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (self.frame().width() / ideal_chunk_size).floor();
        let rows = (self.frame().height() / ideal_chunk_size).floor();

        // Reset all chunks.
        for chunk in self.chunks.iter_mut() {
            chunk.clear()
        }

        // Grow the grid if needed (eg. if the frame size increases)
        self.chunks
            .resize_with_defaults(columns as usize, rows as usize);

        let frame = Frame::new(0., 0., screen_width(), screen_height());
        // Register all particles within their current chunks.
        for particle in &self.particles {
            if let Some(chunk) = self.chunks.get_mut_by_pos(particle.pos, frame) {
                chunk.push(particle.index);
            }
        }
    }

    fn update_densities(&mut self) {
        let particles = &self.particles;

        self.densities
            .par_iter_mut()
            .zip(self.pressures.par_iter_mut())
            .enumerate()
            .for_each(|(i, (density, pressure))| {
                let mut local_density = 0.0;

                for j in 0..particles.len() {
                    let displacement = particles[i].pos - particles[j].pos;
                    let distance = displacement.length();

                    if distance < SMOOTHING_RADIUS {
                        local_density += Self::smoothing_kernel(distance);
                    }
                }

                *density = local_density;
                *pressure = GAS_CONSTANT * (local_density - REST_DENSITY);
                // *pressure = (GAS_CONSTANT * (local_density - REST_DENSITY)).max(0.0);
            });
    }

    fn update_forces(&mut self) {
        let particles = &self.particles;
        let densities = &self.densities;
        let pressures = &self.pressures;
        let chunks = &self.chunks;
        let frame = Frame::new(0., 0., screen_width(), screen_height());

        self.forces
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, force)| {
                let position = particles[i].pos;
                let mut total_force = Vec2::new(0.0, GRAVITY_FACTOR);

                for chunk in chunks.get_neighbourhood_at_pos(position, 1, frame) {
                    for j in chunk.iter().copied() {
                        if i == j {
                            continue;
                        }

                        let pressure = Self::pressure_force(i, j, particles, densities, pressures);

                        total_force += pressure;
                    }
                }

                *force = total_force;
            });
    }

    fn smoothing_kernel(distance: f32) -> f32 {
        // Simple quadratic kernel
        if distance >= SMOOTHING_RADIUS {
            return 0.0;
        }

        (1.0 - distance / SMOOTHING_RADIUS).powi(2)

        // Poly6 Smoothing Kernel
        // Can't use f32::powi(8) here as it is not const
        // const SMOOTHING_RADIUS_POW8: f32 = SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS
        //     * SMOOTHING_RADIUS;

        // const SMOOTHING_CONSTANT: f32 = 4.0 / (std::f32::consts::PI * SMOOTHING_RADIUS_POW8);

        // if distance >= SMOOTHING_RADIUS {
        //     return 0.0;
        // }

        // let smoothing_radius_squared = SMOOTHING_RADIUS.powi(2);
        // let x = smoothing_radius_squared - distance.powi(2);

        // println!("{}", SMOOTHING_CONSTANT);

        // SMOOTHING_CONSTANT * x.powi(3)
    }

    fn pressure_force(
        p1: usize,
        p2: usize,
        particles: &[FluidParticle],
        pressures: &[f32],
        densities: &[f32],
    ) -> Vec2 {
        let particle1 = &particles[p1];
        let particle2 = &particles[p2];
        let pressure1 = pressures[p1];
        let pressure2 = pressures[p2];
        let density2 = densities[p2];
        let density_avg = (densities[p1] + densities[p2]) * 0.5;

        // if density2 == 0.0 {
        //     return Vec2::ZERO;
        // }

        // println!("{}, {}", pressures[p1] + pressures[p2], density_avg);

        let pressure = (pressure1 + pressure2) / (2.0 * density_avg);
        // let pressure = (pressures[p1] + pressures[p2]) / (2.0 * density2);
        // let pressure = pressures[p1] + pressures[p2];

        let displacement = particle2.pos - particle1.pos;
        -pressure * Self::spiky_gradient(displacement)
    }

    fn spiky_gradient(displacement: Vec2) -> Vec2 {
        let distance = displacement.length();
        if distance == 0.0 || distance >= SMOOTHING_RADIUS {
            return Vec2::ZERO;
        }

        let direction = displacement.normalize();
        // let strength = ((SMOOTHING_RADIUS - distance) / SMOOTHING_RADIUS).powi(2);
        let strength = 1.0 - (distance / SMOOTHING_RADIUS);

        direction * strength
    }

    fn frame(&self) -> Frame {
        self.layout.frame
    }
}

impl Update for SimpleFluidSim {
    fn update(&mut self) {
        let update_start = Instant::now();
        let deltatime = update_start - self.last_update;

        self.layout.refresh();
        let frame = self.frame();
        self.update_chunks();
        self.update_densities();
        self.update_forces();

        if is_mouse_button_down(MouseButton::Left) {
            let mouse_pos = Vec2::from(mouse_position());
            for (i, particle) in self.particles.iter().enumerate() {
                let displacement = mouse_pos - self.frame().pos() - particle.pos;

                if (Vec2::from(mouse_pos) - self.frame().pos() - particle.pos).length()
                    < SMOOTHING_RADIUS
                {
                    self.forces[i] -= (displacement / SMOOTHING_RADIUS) * 10000.
                }
            }
        }

        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, particle)| {
                let force = self.forces[i];
                particle.update(deltatime, force, &frame)
            });

        self.last_update = update_start;
    }
}

impl Draw for SimpleFluidSim {
    fn draw(&self) {
        let target = render_target(self.frame().width() as u32, self.frame().height() as u32);
        target.texture.set_filter(FilterMode::Nearest);

        // Draw into the target
        set_camera(&Camera2D {
            render_target: Some(target.clone()),
            zoom: vec2(2.0 / target.texture.width(), -2.0 / target.texture.height()),
            target: vec2(target.texture.width() / 2.0, target.texture.height() / 2.0),
            ..Default::default()
        });

        clear_background(Color::new(0., 0., 0., 0.));
        for particle in &self.particles {
            draw_circle(
                particle.pos.x,
                particle.pos.y,
                30.,
                Color::new(1.0, 1.0, 1.0, 0.05),
            );
        }

        // Draw to the screen
        set_default_camera();

        // Draw the generated texture
        // gl_use_material(&self.draw_material);
        // let offset = crate::OUTLINE_THICKNESS / 2.;
        // draw_texture_ex(
        //     &target.texture,
        //     self.frame().x() + offset,
        //     self.frame().y() + offset,
        //     Color::new(0., 0., 0., 0.),
        //     DrawTextureParams {
        //         dest_size: Some(vec2(
        //             self.frame().width() - offset * 2.,
        //             self.frame().height() - offset * 2.,
        //         )),
        //         flip_y: true,
        //         ..Default::default()
        //     },
        // );
        // gl_use_default_material();

        let mouse_pos = Vec2::from(mouse_position());
        for particle in &self.particles {
            if (Vec2::from(mouse_pos) - self.frame().pos() - particle.pos).length()
                < SMOOTHING_RADIUS
            {
                draw_circle(
                    particle.pos.x + self.frame().x(),
                    particle.pos.y + self.frame().y(),
                    3.,
                    Color::new(0.8, 0.2, 0.2, 0.6),
                );
            } else {
                draw_circle(
                    particle.pos.x + self.frame().x(),
                    particle.pos.y + self.frame().y(),
                    3.,
                    Color::new(0.2, 0.6, 0.8, 0.6),
                );
            }
        }
    }
}
