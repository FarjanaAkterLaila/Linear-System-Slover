import tkinter as tk
from tkinter import scrolledtext
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt


def validate_solution(coefficients, constants, solution):
    equations_satisfied = np.allclose(coefficients.dot(solution), constants)
    return equations_satisfied

def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.concatenate((A, b.reshape(-1, 1)), axis=1)
    steps = [augmented_matrix.copy()]

    for i in range(n):
        max_index = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_index]] = augmented_matrix[[max_index, i]]

        pivot = augmented_matrix[i, i]
        if pivot == 0:
            continue
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / pivot
            augmented_matrix[j] -= factor * augmented_matrix[i]

        steps.append(augmented_matrix.copy())

    for i in range(n - 1, -1, -1):
        augmented_matrix[i] /= augmented_matrix[i, i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

        steps.append(augmented_matrix.copy())

    return steps

def matrix_inversion(A, b):
    try:
        A_inv = inv(A)
        solution = np.dot(A_inv, b)
        return solution, A_inv
    except np.linalg.LinAlgError:
        return None, None

def jacobi_iteration(A, b, iterations, tolerance):
    n = len(b)
    x = np.zeros(n)
    steps = []
   
    for itr in range(iterations):
        x_new = np.zeros(n)
        
        for i in range(n):
            sum_val = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_val) / A[i, i]
            print(np.linalg.norm(x_new - x))
        # Check convergence using tolerance
        if np.linalg.norm(x_new - x) < tolerance:
            
            validation_result = validate_solution(A, b, x_new)
            print(validation_result)
            # if validation_result:
            #     break  # Exit the loop if the solution is validated
            # else:
            #     scrolled_text.config(state=tk.NORMAL)
            #     scrolled_text.insert(tk.END, f"Iteration {itr+1}: Solution does not satisfy the original equations.\n")
            #     scrolled_text.config(state=tk.DISABLED)
        x = x_new.copy()
        steps.append(x.copy())

    return x, steps

def gauss_seidel_iteration(A, b, iterations, tolerance):
    n = len(b)
    x = np.zeros(n)
    steps = []

    for itr in range(iterations):
        x_new = np.zeros(n)
        for i in range(n):
            sum_val = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_val) / A[i, i]
        
        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new.copy()
        steps.append(x.copy())

    return x, steps

def display_gaussian_steps(steps , variable_names):
    # Clear the scrolled text widget
    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.delete(1.0, tk.END)
    scrolled_text.config(state=tk.DISABLED)

    # Display the steps in the scrolled text widget for Gaussian Elimination
    result_text = "Gaussian Elimination Steps:\n"

    for i, step in enumerate(steps):
        result_text += f"\nStep {i + 1}:\n"
        for row_values in step:
            result_text += " | ".join([f"{value:.4f}" for value in row_values]) + " |\n"
        result_text += "\n"

    result_text += "\nResults:\n"
    for i, variable in enumerate(variable_names):
        result_text += f"{variable} = {steps[-1][i, -1]:.4f}\n"

    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.insert(tk.END, result_text)
    scrolled_text.yview(tk.END) 
    scrolled_text.config(state=tk.DISABLED)  # Disable editing
    
def display_inversion_step(solution, A_inv,original_matrix, variable_names):
    try:
        # print(original_matrix)
        # Calculate the determinant of the original matrix A
        det_A = np.linalg.det(original_matrix)
        
        # Display the steps in the scrolled text widget for Matrix Inversion
        result_text = "Matrix Inversion Steps:\n"
        result_text += f"\nDeterminant of the original matrix A: {det_A:.4f}\n"
        result_text += f"\nInverted Matrix:\n{A_inv}\n"
        result_text += "\nResults:\n"
        for i, variable in enumerate(variable_names):
            result_text += f"{variable} = {solution[i]:.4f}\n"

        scrolled_text.config(state=tk.NORMAL)
        scrolled_text.delete(1.0, tk.END)
        scrolled_text.insert(tk.END, result_text)
        scrolled_text.yview(tk.END)  # Auto-scroll to the end
        scrolled_text.config(state=tk.DISABLED)

    except np.linalg.LinAlgError:
        scrolled_text.config(state=tk.NORMAL)
        scrolled_text.insert(tk.END, "Matrix A is singular. Unable to calculate determinant and perform inversion.\n")
        scrolled_text.yview(tk.END)  # Auto-scroll to the end
        scrolled_text.config(state=tk.DISABLED)

def display_iteration_header(headers):
    result_text = "  Iteration Steps:\n"
    result_text += "-" * len("Iteration Steps:") * 4 + "\n\n"

    column_width = 12  

    # Create the header row
    result_text += f"{'Steps ':>{column_width}} |  " + "  | ".join([f"{h:<{column_width}}" for h in headers]) + "\n\n"

    # Create the separator row with dashes
    result_text += "-" * len("Iteration Steps:") * 4 + "\n\n"

    # Update the scrolled text widget with padding
    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.insert(tk.END, f"\n{' ' * column_width}\n{result_text}")
    scrolled_text.config(state=tk.DISABLED)

def display_iteration_steps_with_delay(steps, method_headers):
    headers_displayed = False

    def display_next_step(step_index):
        nonlocal headers_displayed
        if step_index < len(steps):
            if not headers_displayed:
                display_iteration_header(method_headers)
                headers_displayed = True

            display_iteration_step(step_index + 1, steps[step_index])
            root.after(500, display_next_step, step_index + 1)
        else:
            # Display the final result after all steps
            display_final_result(steps[-1],  method_headers)

    # Start displaying steps with delay
    display_next_step(0)
def generate_default_variable_names(num_variables):
    variable_names = [f"x{i+1}" for i in range(num_variables)]
    return variable_names
def display_final_result(final_result, variable_names):
    result_text = "\nFinal Result:\n\n"
    for i, variable in enumerate(variable_names):
        result_text += f"{variable} = {final_result[i]:.4f}\n"

    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.insert(tk.END, result_text)
    scrolled_text.config(state=tk.DISABLED)
    
def display_iteration_step(iteration, step):
    result_text = "  Iteration Steps:\n"
    result_text += "-" * len("Iteration Steps:") * 4 + "\n\n"

    column_width = 12  # Adjust the width based on your preference

    # Display the current iteration step
    result_text += f"{iteration:^{column_width}.0f} | "
    result_text += " | ".join([f"{val:>{column_width}.3f}" for val in step]) + "\n\n"

    # Update the scrolled text widget with padding
    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.insert(tk.END, result_text)
    scrolled_text.yview(tk.END)  # Auto-scroll to the end
    scrolled_text.config(state=tk.DISABLED)
def calculate_individual_errors(steps):
    errors = [[norm(steps[i][j] - steps[i - 1][j]) for j in range(len(steps[0]))] for i in range(1, len(steps))]
    return errors

def display_error_table(steps,variable_names):
    errors = calculate_individual_errors(steps)

    result_text = "\nError Estimation Table:\n\n"
    result_text += "Iteration |  "
    result_text += " | ".join([f"Error ({var})" for var in variable_names]) + "\n"
    result_text += "-" * 50 + "\n"

    for i, error in enumerate(errors, start=1):
        result_text += f"{i:^9} | "
        result_text += " | ".join([f"{err:.6f}" for err in error]) + "\n"

    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.insert(tk.END, result_text)
    scrolled_text.config(state=tk.DISABLED)

def display_error_graph(steps):
    errors = calculate_individual_errors(steps)
    iterations = range(1, len(steps))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o', linestyle='-')
    plt.title('Error Estimation Graph')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def error_button_click():
    method = method_var.get()

    if method in ["Jacobi Iteration", "Gauss-Seidel Iteration"]:
        coefficients_input = entry_coefficients.get("1.0", tk.END).strip()
        constants_input = entry_constants.get("1.0", tk.END).strip()

        coefficients_rows = [row.split() for row in coefficients_input.splitlines()]
        coefficients = np.array([[float(val) for val in row] for row in coefficients_rows])
        num_variables = len(coefficients[0])
        variable_names = generate_default_variable_names(num_variables)
        print(variable_names)
        constants = np.array([float(val) for val in constants_input.split()])

        iterations = int(iteration_entry.get()) if method in ["Jacobi Iteration", "Gauss-Seidel Iteration"] else None
        tolerance = float(tolerance_entry.get()) if method in ["Jacobi Iteration", "Gauss-Seidel Iteration"] else None
        if iterations is not None:
            if method == "Jacobi Iteration":
                solution, steps = jacobi_iteration(coefficients.copy(), constants.copy(), iterations, tolerance)
               
                display_error_table(steps,variable_names)
                display_error_graph(steps)
            elif method == "Gauss-Seidel Iteration":
                solution, steps = gauss_seidel_iteration(coefficients.copy(), constants.copy(), iterations, tolerance)
                display_error_table(steps,variable_names)
                display_error_graph(steps)
        else:
            scrolled_text.config(state=tk.NORMAL)
            scrolled_text.insert(tk.END, "Please enter the number of iterations.")
            scrolled_text.config(state=tk.DISABLED)



def solve_system():
    try:
         # Get coefficients and constants from user inputs
        coefficients_input = entry_coefficients.get("1.0", tk.END).strip()
        constants_input = entry_constants.get("1.0", tk.END).strip()

        coefficients_rows = [row.split() for row in coefficients_input.splitlines()]
        coefficients = np.array([[float(val) for val in row] for row in coefficients_rows])

        constants = np.array([float(val) for val in constants_input.split()])

        # Generate default variable names based on the number of coefficients
        num_variables = len(coefficients[0])
        variable_names = generate_default_variable_names(num_variables)

        method = method_var.get()
        solution = None
        steps = None
       
        scrolled_text.config(state=tk.NORMAL)
        scrolled_text.delete(1.0, tk.END)

        if method == "Gaussian Elimination":
            steps = gaussian_elimination(coefficients.copy(), constants.copy())
            solution = steps[-1][:, -1]
            display_gaussian_steps(steps, variable_names)
            validation_result = validate_solution(coefficients, constants, solution)
            print(validation_result)
            if validation_result:
                scrolled_text.config(state=tk.NORMAL)
                scrolled_text.insert(tk.END, "\nSolution validated: The results satisfy the original equations.\n")
                scrolled_text.config(state=tk.DISABLED)
                
            else:
                 scrolled_text.config(state=tk.NORMAL)
                 scrolled_text.insert(tk.END, "\nSolution invalid: The results do not satisfy the original equations.\n")
                 scrolled_text.config(state=tk.DISABLED)
                
            
        elif method == "Matrix Inversion":
             solution, A_inv = matrix_inversion(coefficients.copy(), constants.copy())
             display_inversion_step(solution, A_inv, coefficients, variable_names)
             validation_result = validate_solution(coefficients, constants, solution)
             print(validation_result)
             if validation_result:
                scrolled_text.config(state=tk.NORMAL)
                scrolled_text.insert(tk.END, "\nSolution validated: The results satisfy the original equations.\n")
                scrolled_text.config(state=tk.DISABLED)
             else:
                 scrolled_text.config(state=tk.NORMAL)
                 scrolled_text.insert(tk.END, "\nSolution invalid: The results do not satisfy the original equations.\n")
                 scrolled_text.config(state=tk.DISABLED)
        elif method in ["Jacobi Iteration", "Gauss-Seidel Iteration"]:
            iterations = int(iteration_entry.get()) if method in ["Jacobi Iteration", "Gauss-Seidel Iteration"] else None
            tolerance = float(tolerance_entry.get()) if iterations else None
            if iterations is not None:
                if method == "Jacobi Iteration":
                    solution, steps = jacobi_iteration(coefficients.copy(), constants.copy(), iterations, tolerance)
                    display_iteration_steps_with_delay(steps, variable_names)
                   
                elif method == "Gauss-Seidel Iteration":
                    solution, steps = gauss_seidel_iteration(coefficients.copy(), constants.copy(), iterations, tolerance)
                    display_iteration_steps_with_delay(steps, variable_names)
            else:
                scrolled_text.insert(tk.END, "Please enter the number of iterations.")

    except ValueError:
        scrolled_text.insert(tk.END, "Please enter valid numeric values.")
    finally:
        scrolled_text.config(state=tk.DISABLED)
       

    
def show_hide_iterations(*args):
    scrolled_text.config(state=tk.NORMAL)
    scrolled_text.delete(1.0, tk.END)
    scrolled_text.config(state=tk.DISABLED)

    if method_var.get() in ["Jacobi Iteration", "Gauss-Seidel Iteration"]:
        iterations_label.pack_forget()
        iteration_entry.pack_forget()
        tolerance_label.pack_forget()
        tolerance_entry.pack_forget()

        solve_button.pack_forget()
        scrolled_text.pack_forget()
        error_button.pack_forget()
        
        iterations_label.pack(pady=(0, 0))
        iteration_entry.pack(pady=(0, 5))

        tolerance_label.pack(pady=(0, 0), padx=10)
        tolerance_entry.pack(pady=(0, 5), padx=10)
        
        solve_button.pack(pady=(0, 10))
        scrolled_text.pack(pady=(0,5))
        error_button.pack(pady=(0, 10))
    else:
        iterations_label.pack_forget()
        iteration_entry.pack_forget()
        tolerance_label.pack_forget()
        tolerance_entry.pack_forget()

        solve_button.pack(pady=(20, 20))

root = tk.Tk()
root.title("Linear System Solver")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = int(screen_width * 0.9)
window_height = int(screen_height * 0.9)
window_x = (screen_width - window_width) // 2
window_y = (screen_height - window_height) // 10

root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
root.configure(bg="skyblue")

title_label = tk.Label(root, text="Linear System Solver", font=("Helvetica", 20, "bold"), anchor="center", padx=10, pady=5)
title_label.pack(fill="x", pady=(0, 15))

label_method = tk.Label(root, text="Select solution method:", bg="skyblue")
label_method.pack()

method_var = tk.StringVar(root)
method_var.set("Gaussian Elimination")
methods = ["Gaussian Elimination", "Matrix Inversion", "Jacobi Iteration", "Gauss-Seidel Iteration"]

dropdown = tk.OptionMenu(root, method_var, *methods, command=show_hide_iterations)
dropdown.pack(pady=(0, 15))

show_iterations_field = False

iterations_label = tk.Label(root, text="Enter number of max iterations:", bg="skyblue")
iteration_entry = tk.Entry(root)

# Create input fields for tolerance and maximum iterations
tolerance_label = tk.Label(root, text="Enter convergence tolerance:", bg="skyblue")
tolerance_entry = tk.Entry(root)


label_coefficients = tk.Label(root, text="Enter coefficients (separate numbers by space or newline):", bg="skyblue")
label_coefficients.pack(pady=(6, 0))

entry_coefficients = scrolledtext.ScrolledText(root, height=6, width=30, wrap=tk.WORD, fg="black", font=("Helvetica", 12))
entry_coefficients.pack(pady=(0, 10))

label_constants = tk.Label(root, text="Enter constants:", bg="skyblue")
label_constants.pack(pady=(0, 0))

entry_constants = tk.Text(root, height=1, width=30)
entry_constants.pack(pady=(0, 10))

solve_button = tk.Button(root, text="Solve", command=solve_system, bg="green", fg="white")
solve_button.pack(pady=(10, 10))

scrolled_text = scrolledtext.ScrolledText(root, height=10, width=50, wrap=tk.WORD, fg="black", font=("Helvetica", 12))
scrolled_text.pack(pady=(0, 20))
scrolled_text.config(state=tk.DISABLED)

error_button = tk.Button(root, text="Error", command=error_button_click, bg="red", fg="white")
error_button.pack(pady=(0, 5))
root.mainloop()
