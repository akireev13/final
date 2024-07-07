from flask import Flask, render_template, request
import math
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import matplotlib

# Set the backend to Agg
matplotlib.use('Agg')

app = Flask(__name__)

# Functions from your code
def input_func(x):
    return 10 * math.sin(5 * x) + x ** 2 - 2 * x

def interpolation(x, y):
    n = len(x)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i][j] = x[i] ** j

    x_coeff = np.linalg.solve(A, y)

    def P(x):
        ans = 0
        cur = 1
        for coeff in x_coeff:
            ans += coeff * cur
            cur *= x

        return ans

    return P

def approximation(x, y, functions):
    rows = len(x)
    cols = len(functions)
    
    A = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            A[i][j] = functions[j](x[i])
            
    coeffs = np.linalg.solve(A.T @ A, A.T @ y)
    def f(x):
        ans = 0
        for i in range(len(coeffs)):
            ans += coeffs[i] * functions[i](x)
        return ans
    return f

def generate_i_term(i):
    def f(x):
        return x ** i
    return f

def generate_i_term_trig(i):
    def f(x):
        k = (i + 1) // 2    
        if i % 2 == 1:
            return math.sin(k * x)
        return math.cos(k * x)
    return f

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    error_message = None

    if request.method == 'POST':
        try:
            input_type = int(request.form['input_type'])
            m = int(request.form['m'])

            if input_type == 0:
                x = list(map(float, request.form['x_vals'].split()))
                y = list(map(float, request.form['y_vals'].split()))
            else:
                n = int(request.form['n'])
                left = float(request.form['left'])
                right = float(request.form['right'])

                x = np.linspace(left, right, n)
                y = [input_func(i) for i in x]

            fig, ax = plt.subplots()

            if input_type == 1:
                x_space = np.linspace(left, right, 1000)
                y_space = [input_func(i) for i in x_space]
                ax.plot(x_space, y_space, label="Input function")

            ax.plot(x, y, 'o', label="Base points")

            poly_interpol = interpolation(x, y)
            x_interpol = np.linspace(min(x), max(x), 1000)
            y_interpol = [poly_interpol(xi) for xi in x_interpol]
            ax.plot(x_interpol, y_interpol, label="Interpolation polynomial")

            functions = [generate_i_term(i) for i in range(m + 1)]
            poly_approx = approximation(x, y, functions)
            x_approx = np.linspace(min(x), max(x), 1000)
            y_approx = [poly_approx(xi) for xi in x_approx]
            ax.plot(x_approx, y_approx, label="Approximation polynomial")

            functions_general = [generate_i_term_trig(i) for i in range(2 * m + 1)]
            poly_approx_general = approximation(x, y, functions_general)
            x_approx_general = np.linspace(min(x), max(x), 1000)
            y_approx_general = [poly_approx_general(xi) for xi in x_approx_general]
            ax.plot(x_approx_general, y_approx_general, label="Approximation general polynomial")

            ax.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            plot_url = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:
            error_message = f'Error: {str(e)}'

    return render_template('index.html', plot_url=plot_url, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
