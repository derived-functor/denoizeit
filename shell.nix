{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; } }:

let
  cuda = pkgs.cudaPackages_12_8;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg_7
    portaudio
    stdenv.cc.cc.lib
    zlib
    cuda.cudatoolkit
    cuda.cudnn
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.ffmpeg_7
      pkgs.portaudio
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      cuda.cudatoolkit
      cuda.cudnn
    ]}:$LD_LIBRARY_PATH
    
    export CUDA_PATH=${cuda.cudatoolkit}
    
    echo "CUDA 12.8 environment loaded."
    exec ${pkgs.fish}/bin/fish
  '';
}
