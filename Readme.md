# LDA on Design Issues

This repository contains the code and resources for Assignment 2 of the DSSE project.

### Generating the `accelerator` Python Module

Follow these steps to generate the `accelerator` Python module needed for text processing. This involves cloning the repository, setting up the environment, and building the Rust extension.

#### Prerequisites
- **Rust**: Ensure Rust is installed on your system. Install it from [rust-lang.org](https://www.rust-lang.org/learn/get-started).
- **Python**: Ensure you have Python installed (compatible with the Rust setup).

#### Steps to Generate the `accelerator` Module

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/mining-design-decisions/mining-design-decisions.git
   cd mining-design-decisions/deep_learning
   ```

2. **Set Up Python Environment**:
   - Install the necessary Python packages:
     ```sh
     pip install setuptools_rust
     ```

3. **Build the Rust Extension**:
   - The `setup.py` file is already present in the `deep_learning` directory. Run the following command to compile the Rust code and generate the Python module:
     ```sh
     python setup.py build_ext --inplace
     ```

4. **Move the Compiled Module**:
   - Move the generated `.pyd` (Windows) or equivalent shared library file to the `lib` directory of your project:
     ```sh
     mkdir -p ../lib
     mv accelerator*.pyd ../lib/  # Adjust for your OS if necessary (e.g., .so for Linux, .dylib for macOS)
     ```

5. **Update Configuration**:
   - Ensure `config.py` contains the correct path to the `accelerator` module:
     ```python
     # ... Other config variables
     
     ACCELERATOR_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'accelerator.cp310-win_amd64.pyd')
     ```