import os
import re
import inspect
import importlib
import sys

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath('.'))

def scan_env_files(directory):
    """Scan Python files in a directory for environment classes with register_env decorator"""
    env_info = []
    
    # Get all Python files in the directory
    py_files = [f for f in os.listdir(directory) if f.endswith('.py') and not f.startswith('__')]
    
    for py_file in py_files:
        module_name = f"mani_skill.envs.tasks.coin_bench.{py_file[:-3]}"
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Inspect the module for classes
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    # Check if the class has a register_env decorator
                    source = inspect.getsource(obj)
                    register_match = re.search(r'@register_env\("([^"]+)"', source)
                    if register_match:
                        env_id = register_match.group(1)
                        
                        # Extract class docstring for description
                        description = obj.__doc__.strip() if obj.__doc__ else "No description available"
                        description = description.split('\n')[0] if description else "No description available"
                        
                        env_info.append({
                            'env_id': env_id,
                            'class_name': name,
                            'file': py_file,
                            'description': description
                        })
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    return env_info

def update_markdown(md_file, env_info):
    """Update the markdown file with environment information"""
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Create a new table for the coin_bench environments
    table = "| Environment ID | Class Name | Source File | Description |\n"
    table += "|----------------|------------|-------------|-------------|\n"
    
    for env in env_info:
        table += f"| `{env['env_id']}` | {env['class_name']} | {env['file']} | {env['description']} |\n"
    
    # Add the new table to the markdown file
    new_section = "\n## Coin Bench Environments (Auto-Generated)\n\n"
    new_section += table
    
    # Append the new section to the end of the file
    updated_content = content + new_section
    
    with open(md_file, 'w') as f:
        f.write(updated_content)

if __name__ == "__main__":
    coin_bench_dir = "mani_skill/envs/tasks/coin_bench"
    md_file = "SupportedEnv.md"
    
    env_info = scan_env_files(coin_bench_dir)
    update_markdown(md_file, env_info)
    
    print(f"Found {len(env_info)} environments:")
    for env in env_info:
        print(f"- {env['env_id']} ({env['class_name']})")
