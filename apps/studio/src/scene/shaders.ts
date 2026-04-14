// ---------- vertex / fragment shaders for neuron point cloud ----------

export const neuronVertexShader = /* glsl */ `
  uniform float uTime;
  attribute float aIndex;
  attribute float aBaseActivation;
  attribute float aIsAttn;
  attribute float aSelected;

  varying float vActivation;
  varying float vIsAttn;
  varying float vSelected;

  void main() {
    float posX = position.x;
    float time = uTime;
    
    // sweeping animations computed on GPU
    float sweepBase = sin(-posX * 0.3 + time * 3.0) * 0.5 + 0.5;
    float sweepPulse = sin(-posX * 0.8 + time * 8.0) * 0.5 + 0.5;
    
    float ambientGlow = pow(sweepBase, 4.0) * 0.15;
    float ripple = pow(sweepPulse, 32.0);
    
    float starryTwinkle = (sin(time * 2.5 + posX * 1.5 + aIndex * 0.1) * 0.5 + 0.5) * 0.08;
    
    float act = abs(aBaseActivation);
    float dynamicAct = act + ambientGlow + (ripple * 2.2 * max(0.12, act)) + starryTwinkle;
    
    float sizeBase = 0.012 + mod(aIndex, 8.0) * 0.005;
    float sizeMod = pow(abs(dynamicAct), 1.5) * 0.28;
    float finalSize = min(sizeBase + sizeMod, 0.65);

    vActivation = dynamicAct;
    vIsAttn = aIsAttn;
    vSelected = aSelected;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = finalSize * (220.0 / -mvPosition.z);
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

    // Color mapping: True Milky Way (Deep cosmic dust, cold icy blue, brilliant golden core)
    vec3 voidSpace = vec3(0.015, 0.04, 0.1);     // soft dark blue nebula base (not pitch black)
    vec3 iceBlue = vec3(0.35, 0.8, 1.0);         // ambient cold blue starlight
    vec3 whiteCore = vec3(0.96, 0.98, 1.0);      // high energy
    vec3 warmGold  = vec3(1.0, 0.98, 0.75);      // golden/warm #fffae6 ignition
    vec3 pureWhite = vec3(1.0, 1.0, 1.0);        // ultimate pulse clip

    vec3 hotPink = vec3(1.0, 0.15, 0.45);        // structural systems/heads
    vec3 yellowCore = vec3(1.0, 0.7, 0.1);
    vec3 selectLime = vec3(0.6, 1.0, 0.1);

    vec3 color;
    if (vSelected > 0.5) {
      color = mix(selectLime, pureWhite, clamp(act * 0.5, 0.0, 1.0));
    } else if (vIsAttn > 0.5) {
      color = mix(hotPink, yellowCore, clamp(act * 1.8, 0.0, 1.0));
      color = mix(voidSpace, color, clamp(act * 4.0, 0.0, 1.0));
    } else {
      // Very gradual transition from dark void to brilliant stars
      color = mix(voidSpace, iceBlue, smoothstep(0.005, 0.15, act));
      color = mix(color, whiteCore, smoothstep(0.15, 0.4, act));
      color = mix(color, warmGold, smoothstep(0.4, 0.8, act));
      color = mix(color, pureWhite, smoothstep(0.8, 1.3, act)); // Explosive pulse
    }

    // Inactive dust has 12% alpha so the galaxy structure is visible. A pulse/activation rapidly pushes it up
    float baseAlpha = mix(0.12, 1.0, smoothstep(0.005, 0.4, act));
    float alpha = baseAlpha * glow * 1.8;
    if (vSelected > 0.5) alpha = max(alpha, 0.85);

    // Multiplicative glow logic to bloom the entire scene aggressively on strong pulses
    float safeIntensity = max(intensity, 0.0001);
    float bloomMultiplier = 1.0 + pow(max(0.0, act - 0.2), 1.5) * 6.0; 
    
    gl_FragColor = vec4(color * bloomMultiplier, min(alpha, 1.0));
  }
`;
