{
  description = "Flake for ThriftAttention kernel build";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels?dir=kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
