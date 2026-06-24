{
  description = "Flake for ThriftAttention kernel build";

  inputs = {
    patched-nixpkgs.url = "path:./patched-nixpkgs";
    kernel-builder = {
      url = "github:huggingface/kernels";
      inputs.nixpkgs.follows = "patched-nixpkgs";
    };
  };

  outputs =
    {
      self,
      kernel-builder,
      ...
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
