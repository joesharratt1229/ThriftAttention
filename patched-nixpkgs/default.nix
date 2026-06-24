{ system ? builtins.currentSystem, overlays ? [ ], config ? { }, ... }@args:

let
  upstreamSource = builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs.git";
    rev = "fec2c46cca5bf9767486a290abae51200b656d69";
    allRefs = true;
  };

  # Disable tests on python packages whose test suites have environment-specific
  # races in the enroot-nested sandbox (tmpdir cleanup, file-descriptor exhaustion,
  # etc.). Only includes packages we've actually observed failing — keeping the
  # override list small minimizes cache invalidation downstream.
  skipFlakyTestsOverlay = final: prev: {
    pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
      (python-self: python-super: {
        aiohttp = python-super.aiohttp.overridePythonAttrs (_: {
          doCheck = false;
          doInstallCheck = false;
        });
        fsspec = python-super.fsspec.overridePythonAttrs (_: {
          doCheck = false;
          doInstallCheck = false;
        });
      })
    ];
  };
in
import upstreamSource (args // {
  overlays = [ skipFlakyTestsOverlay ] ++ overlays;
})
