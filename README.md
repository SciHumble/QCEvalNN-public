# QCEvalNN

This package is for my master thesis about quantum convolutional neural networks (QCNN) and classical convolutional neural networks (CCNN). For the simulation of QCNNs is PennyLane utilized and for the classical equivalent is torch used.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SciHumble/QCEvalNN-public.git
cd QCEvalNN-public
```

### 2. Insall Required Packages
```bash
pip install -r requirements.txt
```

### Requirements
Make sure you have installed:
 * Python 3.9 or newer
 * `pip` (Python package installer)

### Commit Rules

To maintain a clear and consistent commit history, please follow these rules when making commits:

1. **Write Clear, Concise Commit Messages:**
   - Use the present tense ("Add feature" not "Added feature").
   - Capitalize the first letter of the commit message.
   - Keep the subject line to 50 characters or less.
   - Separate the subject from the body with a blank line.
   - Use the body to explain **what** and **why** rather than **how**.

2. **Prefix Commit Messages with Context:**
   - Use prefixes to provide context for the changes. Examples:
     - `feat`: A new feature
     - `fix`: A bug fix
     - `docs`: Documentation changes
     - `style`: Code style changes (formatting, missing semi-colons, etc.)
     - `refactor`: Code changes that neither fix a bug nor add a feature
     - `test`: Adding or updating tests
     - `chore`: Changes to the build process or auxiliary tools

3. **Examples of Good Commit Messages:**
   - `feat: add support for quantum gates`
   - `fix: resolve issue with state vector calculation`
   - `docs: update README with installation instructions`
   - `style: format code according to PEP-8`
   - `refactor: simplify neural network initialization logic`
   - `test: add unit tests for quantum circuit functions`
   - `chore: update dependency versions`

4. **Make Atomic Commits:**
   - Each commit should represent a single logical change.
   - Avoid mixing unrelated changes in a single commit.

5. **Reference Issues and Pull Requests:**
   - When applicable, reference issues and pull requests in your commit messages. For example, `fix: correct typo in README (closes #42)`.

6. **Review and Squash Commits:**
   - Before merging a pull request, review and squash commits to keep the commit history clean and meaningful.

By following these guidelines, we can ensure that our project's commit history is easy to read and understand, facilitating collaboration and code reviews.

