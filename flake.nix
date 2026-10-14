{
  description = "Development environment for FixRABIES";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    {
      devShells.x86_64-linux.default =
        let
          pkgs = import nixpkgs { system = "x86_64-linux"; };
        in
        pkgs.mkShell {
          packages = [
            pkgs.python314
            pkgs.uv
            pkgs.gcc
          ];
          buildInputs = [ 
            pkgs.python314Packages.pyside6
            pkgs.python314Packages.numpy
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib
            zlib
            glib
          ]);
        };
    };
}
