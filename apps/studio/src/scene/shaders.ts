// ---------- vertex / fragment shaders for neuron point cloud ----------

export const neuronVertexShader = /* glsl */ `
  attribute float aSize;
  attribute float aActivation;
  attribute float aIsAttn;
  attribute float aSelected;

  varying float vActivation;
  varying float vIsAttn;
  varying float vSelected;

  void main() {
    vActivation = aActivation;
    vIsAttn = aIsAttn;
    vSelected = aSelected;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (220.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const neuronFragmentShader = /* glsl */ `
  varying float vActivation;
  varying float vIsAttn;
  varying float vSelected;

  void main() {
    // Circular soft particle
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;

    float glow = 1.0 - smoothstep(0.0, 0.5, dist);

    float act = abs(vActivation);
    float intensity = act * glow;

    // Color mapping: Milky Way palette (Cold blue-white to warm-gold pulses)
    vec3 voidSpace = vec3(0.005, 0.01, 0.02);    // nearly invisible dark star background
    vec3 coldBlueWhite = vec3(0.7, 0.85, 1.0);   // baseline activation
    vec3 warmGold = vec3(1.0, 0.984, 0.902);     // peak flow/pulse (#fffbe6)
    vec3 pureWhite = vec3(1.0, 1.0, 1.0);        // core blinding pulse

    vec3 hotPink = vec3(1.0, 0.2, 0.6);          // attention system
    vec3 sunGold = vec3(1.0, 0.8, 0.2);
    vec3 selectedGreen = vec3(0.5, 1.0, 0.2);

    vec3 color;
    if (vSelected > 0.5) {
      color = mix(selectedGreen, pureWhite, act * 0.4);
    } else if (vIsAttn > 0.5) {
      color = mix(hotPink, sunGold, clamp(act * 1.5, 0.0, 1.0));
      color = mix(voidSpace, color, clamp(act * 3.0 + 0.2, 0.0, 1.0));
    } else {
      // Cosmic optical flow mapping
      color = mix(voidSpace, coldBlueWhite, min(act / 0.15, 1.0));
      color = mix(color, warmGold, smoothstep(0.15, 0.6, act));
      color = mix(color, pureWhite, smoothstep(0.6, 1.2, act));
    }

    // Inactive background stars have exceptionally low alpha, pulses explode in brightness
    float alpha = mix(0.015, 1.0, act) * glow * 2.0;
    if (vSelected > 0.5) alpha = max(alpha, 0.8);

    // Multiplicative exponential glow for massive visual impact during pulse
    gl_FragColor = vec4(color * (1.0 + pow(intensity, 1.6) * 3.5), min(alpha, 1.0));
  }
`;
