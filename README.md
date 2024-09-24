# Bouncing Ball Task

Repo defining the base bouncing ball task used for both computational modeling and 
the human task.

## Getting Started

### Installing Dependencies

Clone and navigate to the project:
```bash
git clone git@github.com:APRashedAhmed/Bouncing-Ball-Task.git
cd Bouncing-Ball-Task
```

Install the conda environment and activate as needed: 

```bash
# Install the desired base environment
conda env create -f environment.yaml

# Activate the environment
conda activate bbt
```

[OPTIONAL] Create an autoenv file so the conda environment is automatically loaded
when you navigate to the directory

```bash
# Copy the template
cp .autoenv.template .autoenv

# Run this command to add a line to your bashrc (or whichever shellrc)
echo "autoenv() { [[ -f \"\$PWD/.autoenv\" ]] && source .autoenv ; } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.local/.bashrc
```

### Importing from `bouncing_ball_task`

Since the project was installed, all its components are importable:

```python
# Import the top-level module
import bouncing_ball_task as bbt

# Import a specific module
from bouncing_ball_task import index

# Import a specific function
from bouncing_ball_task.utils.taskutils import last_visible_color
```
