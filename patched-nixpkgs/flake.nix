{
  description = "Local nixpkgs wrapper that disables aiohttp tests";

  inputs.upstream-nixpkgs.url = "github:NixOS/nixpkgs/fec2c46cca5bf9767486a290abae51200b656d69";

  outputs =
    { self, upstream-nixpkgs }:
    {
      lib = upstream-nixpkgs.lib;

      legacyPackages = upstream-nixpkgs.lib.genAttrs upstream-nixpkgs.lib.systems.flakeExposed (
        system: import self { inherit system; }
      );
    };
}
