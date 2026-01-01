import os
import json
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="NexaLang Package Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("name", help="Project name")
    
    add_parser = subparsers.add_parser("add", help="Add a dependency")
    add_parser.add_argument("path", help="Path to local package")
    
    args = parser.parse_args()
    
    if args.command == "init":
        config = {
            "name": args.name,
            "version": "0.1.0",
            "main": "src/main.nxl",
            "dependencies": {}
        }
        with open("nexa.json", "w") as f:
            json.dump(config, f, indent=4)
        os.makedirs("src", exist_ok=True)
        with open("src/main.nxl", "w") as f:
            f.write('fn main() -> i32 {\n    print("Hello, Nexa!");\n    return 0;\n}\n')
        print(f"Project '{args.name}' initialized.")
        
    elif args.command == "add":
        if not os.path.exists("nexa.json"):
            print("Error: No nexa.json found. Run 'nxpkg init' first.")
            return
            
        with open("nexa.json", "r") as f:
            config = json.load(f)
            
        pkg_name = os.path.basename(args.path)
        config["dependencies"][pkg_name] = args.path
        
        with open("nexa.json", "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"Added dependency '{pkg_name}' from '{args.path}'.")

if __name__ == "__main__":
    main()
