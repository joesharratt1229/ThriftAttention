{ system ? builtins.currentSystem, overlays ? [ ], config ? { }, ... }@args:

let
  upstreamSource = builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs.git";
    rev = "fec2c46cca5bf9767486a290abae51200b656d69";
    allRefs = true;
  };

  aiohttpFixOverlay = final: prev: {
    pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
      (python-self: python-super: {
        aiohttp = python-super.aiohttp.overridePythonAttrs (_: {
          doCheck = false;
          doInstallCheck = false;
        });
      })
    ];
  };
in
import upstreamSource (args // {
  overlays = [ aiohttpFixOverlay ] ++ overlays;
})
