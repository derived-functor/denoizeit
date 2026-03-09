{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs;[
    ffmpeg_7
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (with pkgs;[
      ffmpeg_7
      stdenv.cc.cc.lib
      zlib
    ])}:$LD_LIBRARY_PATH
  '';
}
