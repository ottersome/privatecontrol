{ pkgs ? import <nixpkgs> {} }:

let
  poetry2nix = pkgs.poetry2nix;
  python3 = pkgs.python3;
  env = poetry2nix.mkPoetryEnv {
    python = python3;
    poetrylock = ./poetry.lock;
  };
in
pkgs.stdenv.mkDerivation {
  name = "env-test";
  buildInputs = [ env ];
  buildCommand = ''
    ${env}/bin/python -c 'import alembic'
    touch $out
  '';
}

