use macroquad::prelude::*;
use miniquad::graphics::*;

pub fn liquid_material() -> Material {
    load_material(
        ShaderSource::Glsl {
            vertex: DEFAULT_VERTEX_SHADER,
            fragment: LIQUID_FRAGMENT_SHADER,
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
    .unwrap()
}

/// Macroquad's default vertex shader.
pub const DEFAULT_VERTEX_SHADER: &str = r#"
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

/// Assumes that the liquid is drawn using wide, low-alpha particles.
const LIQUID_FRAGMENT_SHADER: &str = r#"
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
    // float alpha = smoothstep(0.0, 1.0, density);
    float surface = 0.35;
    float alpha = smoothstep(surface - 0.05, surface + 0.05, density);

    // Estimate pixel "direction" for normal mapping
    float dx =
        sample_density(vec2(px,0.0)) -
        sample_density(vec2(-px,0.0));
    float dy =
        sample_density(vec2(0.0,px)) -
        sample_density(vec2(0.0,-px));

    // vec2 normal = normalize(vec2(dx, dy));
    vec3 normal = normalize(vec3(dx, dy, 1.0));

    // vec2 light_dir = normalize(vec2(-0.5, -1.0));
    vec3 light_dir = normalize(vec3(-0.5, -0.5, 1.0));

    float diffuse = dot(normal, light_dir);
    diffuse = clamp(diffuse, 0.0, 1.0);

    color *= 0.6 + diffuse * 0.4;
    color *= 0.7 + diffuse * 0.3;

    // Rim highlight
    float edge = length(vec2(dx, dy));
    edge = smoothstep(0.0, 0.6, edge);
    color += vec3(0.3,0.5,1.0) * edge * 3.0;

    gl_FragColor = vec4(
        color[0],
        color[1],
        color[2],
        alpha
    );
}
"#;
