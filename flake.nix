{
  description = "Application packaged using poetry2nix";

  inputs = { flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        # pkgs = nixpkgs.legacyPackages.${system};
        pkgs = import nixpkgs { 
          config.allowUnfree = true;
          cudaSupport = true;
          inherit 
            system;
        };

      in
      {

      # Thisis the only thing that can make my environment run.
      devShells.default = (pkgs.buildFHSUserEnv {
        name = "cuda-env";
          targetPkgs = pkgs: with pkgs; [ 
            # python310Packages.pyqt5
            git
            gitRepo
            gnupg
            autoconf
            curl
            procps
            gnumake
            util-linux
            m4
            gperf
            unzip
            lazygit
            cudatoolkit
            linuxPackages.nvidia_x11
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5
            stdenv.cc
            binutils
            poetry
            qt5.qttools.dev 
            libsForQt5.qt5.qtbase
            python310Packages.python

            # We aslo include here a GUI backend for pyplot Stuff like Qt5Agg
            glib
            fontconfig
            xorg.libX11
            libxkbcommon
            freetype
            dbus

            xorg.libxcb
            xorg.xcbutilwm
            xorg.xcbutilimage
            xorg.xcbutil
            xorg.xcbutilkeysyms
            xorg.xcbutilrenderutil
            stdenv.cc.cc
        ];
        multiPkgs = pkgs: with pkgs; [ zlib ];
        runScript = "zsh";
        # If host has nvim installed, mount it to the container
        profile = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export RPROMPT="%F{cyan}( Poetry)%f"
          export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms";
          # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
          QT_QPA_PLATFORM_PLUGIN_PATH=“${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins”;
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export QT_PLUGIN_PATH=venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins
          export EXTRA_CCFLAGS="-I/usr/include"
          # Tell poetry to use the right python
          # Also mount cache for nvim 
          # Check if poetry environment must be installed for first time 
          poetry_envs=$(poetry env list -q)
          if [ $? -ne 0 ]; then
            poetry env use ${pkgs.python310}/bin/python
            poetry install --no-root
          else # just install poetry shell
            poetry env use ${pkgs.python310}/bin/python
            echo -e "\033[1;33m Remember to use \`poetry shell\` to activate the environment\033[0m"
          fi
          '';
        }).env;

      });
}
