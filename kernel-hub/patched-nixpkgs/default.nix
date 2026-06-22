{ system ? builtins.currentSystem, overlays ? [ ], config ? { }, ... }@args:

let
  upstreamSource = builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs.git";
    rev = "fec2c46cca5bf9767486a290abae51200b656d69";
    allRefs = true;
  };

  # Disable doCheck on every python package that supports overridePythonAttrs.
  # The enroot-nested sandbox + /raid-backed build dir trips tmpdir-cleanup
  # races (e.g. aiohttp, fsspec) that nixpkgs CI does not hit. Tests aren't
  # required to validate the build; we only need importable artifacts.
  skipPythonTestsOverlay = final: prev: {
    pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
      (python-self: python-super:
        builtins.mapAttrs (
          _: pkg:
          if (builtins.isAttrs pkg) && (pkg ? overridePythonAttrs) then
            pkg.overridePythonAttrs (_: {
              doCheck = false;
              doInstallCheck = false;
            })
          else
            pkg
        ) python-super
      )
    ];
  };
in
import upstreamSource (args // {
  overlays = [ skipPythonTestsOverlay ] ++ overlays;
})
