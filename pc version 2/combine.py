import os

def combine_text_files(directory):
    """Combines all .txt files in a directory into a single file,
    prefixing each file's content with its filename.

    Args:
        directory (str): The directory to search for .txt files and save the output.
    """
    output_filename = os.path.join(directory, "combined_text_files.txt")
    with open(output_filename, "w") as outfile:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                outfile.write(f"-- Start of file: {filename} --\n")
                try:
                    with open(filepath, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n-- End of file: {filename} --\n\n")
                except Exception as e:
                    outfile.write(f"Error reading file {filename}: {e}\n\n")
    print(f"Successfully combined .txt files into '{output_filename}' in '{directory}'")

def combine_code_files(directory, output_filename="combined_code_files.base"):
    """Combines all code-related files in a directory into a single file,
    prefixing each file's content with its filename.
    Identifies code files by common extensions: .c, .cc, .cpp, .py, .sh, .slurm, .h.

    Args:
        directory (str): The directory to search for code files and save the output.
        output_filename (str, optional): The name of the output file.
                                         Defaults to "combined_code_files.base".
    """
    code_extensions = (".c", ".cc", ".cpp", ".py", ".sh", ".slurm", ".h")
    output_filepath = os.path.join(directory, output_filename)
    with open(output_filepath, "w") as outfile:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(code_extensions):
                filepath = os.path.join(directory, filename)
                outfile.write(f"-- Start of file: {filename} --\n")
                try:
                    with open(filepath, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n-- End of file: {filename} --\n\n")
                except Exception as e:
                    outfile.write(f"Error reading file {filename}: {e}\n\n")
    print(f"Successfully combined code files into '{output_filepath}' in '{directory}'")

if __name__ == "__main__":
    target_directory = "/Users/faizahmad/Desktop/00hpc/hw7"

    combine_text_files(target_directory)
    combine_code_files(target_directory)