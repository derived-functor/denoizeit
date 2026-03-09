{ pkgs ? import <nixpkgs> { } }:

{
  pkgs.mkShell = with pkgs; [
    uv
    python3
    .11
  ];
}
