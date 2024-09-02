{
  description = "Application packaged using poetry2nix";

  inputs = { flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
    };
  };

  outputs = { self, nixpkgs, unstable, flake-utils, poetry2nix } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        unstable = import unstable { inherit system; };
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
        env = poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = {
            my-app = ./src;
          };
          python = pkgs.python310;
          preferWheels = true;
          overrides = poetry2nix.overrides.withDefaults (
            final: prev: {
              cramjam = prev.cramjam.override{
                preferWheel = true;
                nativeBuildInputs = prev.nativeBuildInputs or [ ] ++ [
                  # pkgs.rustPlatform.cargoSetupHook # handles `importCargoLock`
                  pkgs.rustPlatform.maturinBuildHook # cramjame is based on maturin
                ];
              };
              # fsspec = prev.fsspec.overridePythonAttrs (
              #   old: nixpkgs.lib.optionalAttrs (nixpkgs.lib.versionAtLeast old.version "7.0.0") {
              #     buildInputs = old.buildInputs or [ ] ++ [
              #       prev.hatchling
              #     ];
              #   }
              # );
              fsspec = prev.fsspec.override {
                preferWheel = true;
              };
              scikit-base = prev.scikit-base.overridePythonAttrs(old: rec {
                preferWheel = true;
                postInstall = ''
                    echo "Cleaning up __pycache__ directories..."
                    rm -rf $out/lib/python3.10/site-packages/docs

                    # Optionally call the old postInstall hook if it exists
                    ${old.postInstall or ""}
                '';
              });

              #  chineseize-matplotlib = prev.chineseize-matplotlib.override {
              #   preferWheel = true;
              # };
              # };
            }
          );

        };

      in
      {
        devShells.default = pkgs.mkShell {
        buildInputs = [
          pkgs.git
          pkgs.nodejs
          env
        ];

        shellHook = ''
          echo "Welcome to the development shell with Poetry!"
          export MY_ENV_VAR="value"
        '';
      };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          packages = [ pkgs.poetry ];
        };
      });
}
