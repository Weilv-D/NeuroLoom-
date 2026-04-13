export const officialTraces = [
  {
    id: "spiral-2d-mlp",
    family: "mlp",
    label: "Spiral MLP",
    summary: "Forward fan-out, loss anchor, backward pulse, and decision-boundary drift.",
    accent: "electric",
    path: "/traces/spiral-2d-mlp.loomtrace",
    storyTitle: "Follow the decision boundary being woven from two raw coordinates.",
    watchFor: [
      "The brightest pulse shifts from the hidden layer into the output head during forward frames.",
      "The loss frame acts like a hard checkpoint before the backward pulse returns upstream.",
      "Late update frames calm the stage while confidence rises and the decision plane sharpens.",
    ],
    studioTips: [
      "Select `hidden-a` to inspect how one neuron tracks the outer spiral arc.",
      "Scrub around the loss frame to compare forward and backward edge direction changes.",
      "Use PNG export near the final update frames to capture the cleanest composition.",
    ],
  },
  {
    id: "fashion-mnist-cnn",
    family: "cnn",
    label: "Fashion CNN",
    summary: "Stage-by-stage feature compression with feature-map mosaics and classifier lift.",
    accent: "amber",
    path: "/traces/fashion-mnist-cnn.loomtrace",
    storyTitle: "Watch spatial detail collapse into category evidence across stacked convolution stages.",
    watchFor: [
      "Early stages carry wide, soft activation plates that later compress into denser blocks.",
      "The feature-map mosaic becomes more stable as the classifier head gains confidence.",
      "Backward frames reveal which stages are still noisy and which already hold stable evidence.",
    ],
    studioTips: [
      "Select `conv-2` to inspect later-stage texture compression and lower entropy.",
      "Pause on `pool-2` to compare compression metrics against the feature-map matrix.",
      "Export a frame from the classifier chapter for the clearest stage-to-head composition.",
    ],
  },
  {
    id: "tiny-gpt-style-transformer",
    family: "transformer",
    label: "Tiny GPT Transformer",
    summary: "Token rail, attention ribbons, residual stream, and decode stabilization.",
    accent: "lime",
    path: "/traces/tiny-gpt-style-transformer.loomtrace",
    storyTitle: "Track how a short prompt narrows from wide token context into a single next-token decision.",
    watchFor: [
      "Attention ribbons start broad, then tighten toward the most relevant token relationships.",
      "The residual band keeps information visible between attention and MLP sublayers.",
      "Decode frames show confidence climbing while entropy drops in the logits head.",
    ],
    studioTips: [
      "Select `attn` to inspect the attention matrix as it sharpens across decode frames.",
      "Compare `residual` and `logits` selections to see how focus moves from transport to decision.",
      "Use chapter jumps to isolate token rail, attention, and decode-head compositions.",
    ],
  },
  {
    id: "tiny-sota-vit",
    family: "transformer",
    label: "Tiny SOTA ViT",
    summary: "Patch embedding, multihead attention, and linear classifier visualization for Vision." ,
    accent: "electric",
    path: "/traces/tiny-sota-vit.loomtrace",
    storyTitle: "Explore how image patches fuse into semantic meaning.",
    watchFor: [
      "Patch embeddings map dense 2D images to semantic tokens.",
      "Attention layers aggregate these tokens to learn spatial dependencies.",
      "The classification head makes the final decision on image class."
    ],
    studioTips: [
      "Observe how patch embeddings project the input to higher dimensions.",
      "Track the flow of backward propagation to the initial layers.",
      "Inspect the attention weights to see what parts of the image are focused on."
    ]
  },
] as const;
