# Basic Tutorial: Installing a Prolog Compiler and Executing Code

This short documentation explains how to install a Prolog compiler (**SWI-Prolog**) and run Prolog code on **Windows**, **macOS**, and **Linux**.

---

## 1. Installing SWI-Prolog

**Windows**
1. Download the installer from [SWI-Prolog's official website](https://www.swi-prolog.org/download/stable)  
2. Run the installer and follow the setup instructions  
3. (Optional but recommended) Add SWI-Prolog to your system **PATH** for command-line use  

**macOS (Using Homebrew)**
```sh
brew install swi-prolog

**Linux (Debian/Ubuntu)**

sudo apt-get install swi-prolog

---

2. Running Prolog Code
Method 1: Interactive Shell

    Open Terminal (macOS/Linux) or Command Prompt (Windows)

    Type:

       swipl

    The Prolog interpreter starts. Enter queries directly:

            ?- write('Hello, Prolog!'), nl.

            Press Ctrl+D (or type halt.) to exit

Method 2: Executing a Prolog File

    Create a file example.pl with the following code:

            % example.pl
greet :- write('Hello, Prolog!'), nl.

          Run it in the terminal:

          swipl -s example.pl -g greet -t halt

Explanation:

    -s loads the file

    -g runs a goal (here: greet)

    -t halt exits after execution

Method 3: Using an IDE (Optional)

    VS Code with a Prolog extension (recommended for better coding experience)

    SWI-Prolog's built-in editor (XPCE) — launch with:

      swipl-win   # Windows only

---


3. Verifying Installation

       Check the installed version:

       swipl --version

       Expected output:

       SWI-Prolog version 9.x.x
 
---

###Conclusion###

You now have SWI-Prolog installed and can run Prolog code interactively or from files.