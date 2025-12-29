import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import random

import math

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#==============================================================================================================================
#==============================================================================================================================
# Parámetros: demanda, lead times, BoM, costo compra, costo inventario, precio, probabilidad escenario (unifrome)
#==============================================================================================================================
#==============================================================================================================================

def epsilon(time, scenario, seed=42, lb=0.25, ub=0.75):
    random.seed(seed)
    lala = []
    for s in range(scenario):
        periodo = [random.uniform(lb, ub) for t in range(time)]
        lala.append(periodo)
    return lala

# xi
def rand_sum(seed, prod, time, scenario, mu=0, sigma=10):
    random.seed(seed)
    lala = []
    for s in range(scenario):
        escenario_data = []
        for j in range(prod):
            periodo = [random.gauss(mu, sigma) for t in range(time)]
            escenario_data.append(periodo)
        lala.append(escenario_data)
    return lala

#==============================================================================================================================
#==============================================================================================================================

def lead_times(seed, comp, scenarios, time, lb=1, ub=3, det=True):
    random.seed(seed)
    if det:
        lead_time = []
        for i in range(comp):
            periodo = [max(0, int(random.randint(lb, ub))) for t in range(time)]
            lead_time.append(periodo)

    else:
        lead_time = []
        for t in range(time):
            periodo = []
            for i in range(comp):
                lt_i = [max(0, int(random.randint(lb, ub))) for _ in range(scenarios)]
                periodo.append(lt_i)
            lead_time.append(periodo)

    return lead_time

#==============================================================================================================================
# bom(componentes, productos, total de comp utilizadas por producto)
#==============================================================================================================================
def bill_of_materials(comp=9, prod=9, total=3, comparison=True):
    random.seed(42)
    np.random.seed(42)
    if comparison:

        probabilidades = [1 / comp] * comp
        bom_transpuesta = np.random.multinomial(n=total, pvals=probabilidades, size=prod)
        bom = bom_transpuesta.T

        # bom = np.random.randint(0, 2,(comp, prod))
    else:
        
        # prod: 1|2|3|4
        bom = [[1,1,0,0],  # comp 1
               [2,1,1,0],  # comp 2
               [1,1,1,0],  # comp 3
               [0,0,1,1],  # comp 4
               [0,0,0,1]]  # comp 5
    return bom

G = bill_of_materials()
print("BOM:\n", pd.DataFrame(G))
#==============================================================================================================================

def price_set(inf=80, sup=140, step=2):
    return list(range(inf, sup + 1, step))

def parametros(comp, prod, time, scenarios, cc=False, ci=False, prob=False):
    seed = 450
    random.seed(seed)
    if cc:
        cost_compra = [random.randint(10,15) for i in range(comp)]
    #else:
    #    cost_compra = [2,3]
        # ingreso manual


    if ci:
        #cost_inv = [random.randint(3,5) for i in range(comp)]   
        cost_inv = [x * 0.15 for x in cost_compra]
    #else:
    #    cost_inv = [3,3]
        # ingreso manual



    if prob:
        prob_scenarios = [1/scenarios for s in range(scenarios)] # probabilidad uniforme
        return cost_compra, cost_inv, prob_scenarios
    else:
        return cost_compra, cost_inv

#=========================================================================================================================
def Recourse_problem(seed, time, scenarios, show=True): # Modelo Estocástico (Demanda)
    # ========================================================================
    # INSTANCIA A UTILIZAR (Oh et al.; Slama et al.)
    # ========================================================================
    # t = 1:10
    # epsilon = U[0,25;0,75]
    # xi = N[0,2]
    # a = 200
    # b = 1,5
    # L = U[1,3]
    # c = U[10,15]
    # scenarios = 1000
    # ========================================================================
    random.seed(seed)
    A = bill_of_materials()
    comp = len(A)
    prod = len(A[0])

    price = price_set()
    pr = len(price)

    ypsilon = epsilon(time, scenarios)
    delta = rand_sum(seed, prod, time, scenarios)
    a = 250
    b = 1.2
    C, H, pi = parametros(comp, prod, time, scenarios, cc=True, ci=True, prob=True)
    L = lead_times(seed, comp, scenarios, time, det=False)

    if show:
        print("")
        print("==============================================================================================")
        print("")
        print("BOM:\n", pd.DataFrame(A))
        print("==============================================================================================")
        print("")
        print("Costos de compra:\n", C)
        print("==============================================================================================")
        print("")
        print("Costos de inventario:\n", H)
        print("==============================================================================================")
        print("")
        print("Lead times:")
        try:
            # caso determinista: L[i][t]
            if len(L) == len(A) and all(isinstance(row, (list, tuple)) for row in L):
                for i in range(len(L)):
                    fila = "  " + "  ".join(f"{L[i][t]:>3}" for t in range(len(L[0])))
                    print(f"Componente {i+1}:{fila}")
            else:
                # caso estocástico: L[t][i][s] -> mostramos por escenario
                for s in range(scenarios):
                    print("----------------------------------------------------------------------------------------------")
                    print(f"ESCENARIO {s+1}:")
                    for i in range(comp):
                        fila = "  " + "  ".join(f"{L[t][i][s]:>3}" for t in range(time))
                        print(f"Componente {i+1}:{fila}")
        except Exception:
            # fallback: impresión simple
            print(L)
        print("==============================================================================================")
        print("")


    alpha = {} # Todos los pedidos que han llegado hasta el período t
    for i in range(comp):
        for s in range(scenarios):
            for tau in range(time): # Período en que se hace el pedido (t)
                lead_time = L[tau][i][s] 
                
                for t in range(time): # Período en que se revisa si ya llegó
                    if tau + lead_time <= t:
                        alpha[i, tau, t, s] = 1
                    else:
                        alpha[i, tau, t, s] = 0


    m = gp.Model("Modelo ATO")

    
    x = m.addVars(comp, time, vtype=GRB.INTEGER, name="x")
    w = m.addVars(prod, time, pr, vtype=GRB.BINARY, name = "w")
    y = m.addVars(prod, time, scenarios, vtype=GRB.INTEGER, name="y")
    I = m.addVars(comp, time, scenarios, vtype=GRB.INTEGER, name="I", lb=0)


    D_term = {}
    for j in range(prod):
        for t in range(time):
            # expresión del precio seleccionado para (j,t) (es una LinExpr en w)
            P_expr = gp.quicksum(price[p] * w[j, t, p] for p in range(pr))
            for s in range(scenarios):
                D_term[j, t, s] = ypsilon[s][t] * (a - b * P_expr) + delta[s][j][t]


    f = (gp.quicksum(pi[s]*w[j,t,p]*price[p]*y[j,t,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time))
        )

    m.setObjective(f, GRB.MAXIMIZE)

    # restricción de demanda
    m.addConstrs(y[j,t,s] <= D_term[j,t,s] 
                 for j in range(prod) for t in range(time) for s in range(scenarios))

    # restricción de precio: a cada producto se debe asignar un único precio
    m.addConstrs(gp.quicksum(w[j,t,p] for p in range(pr)) == 1 for j in range(prod) for t in range(time))

    # restricción de balance de inventario
    # alpha = 1 si la orden hecha en tau ya llegó en t
    m.addConstrs(
    (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
     gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
    for i in range(comp) for t in range(time) for s in range(scenarios)
    )
    
    # restricción de disponibilidad de inventario
    #m.addConstrs(0 <= I[i,t,s] - gp.quicksum(y[j,t,s]*A[i][j] for j in range(prod)) for i in range(comp) for t in range(time) for s in range(scenarios))

    # restricción de inventario inicial
    m.addConstrs(I[i,0,s] == 0 for i in range(comp) for s in range(scenarios))

    return m, x, w, y, I, A, D_term
#==============================================================================================================================



#=============================================================================================================================
def get_values(var_dict, shape):
    import numpy as np
    vals = np.empty(shape, dtype=float)
    for index in var_dict.keys():
        vals[index] = var_dict[index].X  # valor exacto sin redondeo
    return vals

def mostrar_x(x):
    import pandas as pd
    df = pd.DataFrame(
        x,
        index=[f"Componente {i+1}" for i in range(x.shape[0])],
        columns=[f"t={t+1}" for t in range(x.shape[1])]
    )
    print("==============================================================================================")
    print("VARIABLE x[i,t]:")
    print("----------------------------------------------------------------------------------------------")
    print(df.to_string(index=True))
    print("==============================================================================================")

def mostrar_w(w):
    import pandas as pd
    prod, time, pr = w.shape
    print("==============================================================================================")
    print("VARIABLE w[j,t,p] - Precio seleccionado por producto y periodo:")
    print("----------------------------------------------------------------------------------------------")
    
    # Crear tabla con precios seleccionados
    price = price_set(70, 120)
    precios_seleccionados = np.zeros((prod, time))
    
    for j in range(prod):
        for t in range(time):
            for p in range(pr):
                if w[j, t, p] > 0.5:  # Es binaria, así que si está cerca de 1
                    precios_seleccionados[j, t] = price[p]
                    break
    
    df = pd.DataFrame(
        precios_seleccionados,
        index=[f"Producto {j+1}" for j in range(prod)],
        columns=[f"t={t+1}" for t in range(time)]
    )
    print(df.to_string(index=True))
    print("==============================================================================================")

def mostrar_y(y):
    import pandas as pd
    prod, time, scen = y.shape
    print("VARIABLE y[j,t,s]: (RELAJADA)")

    for s in range(scen):
        df = pd.DataFrame(
            y[:, :, s],
            index=[f"Producto {j+1}" for j in range(prod)],
            columns=[f"t={t+1}" for t in range(time)]
        )
        print("----------------------------------------------------------------------------------------------")
        print(f"Escenario {s+1}:")
        print(df.to_string(index=True))
    print("==============================================================================================")
    
def mostrar_I(I):
    import pandas as pd
    comp, time, scen = I.shape
    print("VARIABLE I[i,t,s]: (RELAJADA)")
    for s in range(scen):
        df = pd.DataFrame(
            I[:, :, s],
            index=[f"Componente {i+1}" for i in range(comp)],
            columns=[f"t={t+1}" for t in range(time)]
        )
        print("----------------------------------------------------------------------------------------------")
        print(f"Escenario {s+1}:")
        print(df.to_string(index=True))
    print("==============================================================================================")

def mostrar_D(D_term, prod, time, scen):
    import pandas as pd
    import numpy as np
    print("DEMANDA D[j,t,s] (evaluada):")
    D = np.zeros((prod, time, scen), dtype=float)
    for j in range(prod):
        for t in range(time):
            for s in range(scen):
                expr = D_term[j, t, s]
                try:
                    # Si es LinExpr que depende de variables, getValue() devuelve el valor numérico tras optimización
                    val = expr.getValue()
                except Exception:
                    try:
                        val = float(expr)
                    except Exception:
                        val = np.nan
                D[j, t, s] = val

    for s in range(scen):
        df = pd.DataFrame(
            D[:, :, s],
            index=[f"Producto {j+1}" for j in range(prod)],
            columns=[f"t={t+1}" for t in range(time)]
        )
        print("----------------------------------------------------------------------------------------------")
        print(f"Escenario {s+1}:")
        print(df.to_string(index=True))
    print("==============================================================================================")

   
def solver(seed, time, scenarios):
    m, x_vars, w_vars, y_vars, I_vars, A, D_term = Recourse_problem(seed, time, scenarios)
    comp = len(A)
    prod = len(A[0])
    price = price_set()  # Agregar esta línea
    pr = len(price)             # Agregar esta línea
    
    m.setParam('TimeLimit', 600)
    m.setParam('OutputFlag', 1)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        valor_optimo = m.objVal
        print(f"\n Valor optimo: {valor_optimo:.5f}")
        x = get_values(x_vars, (comp, time))
        w = get_values(w_vars, (prod, time, pr))  # Cambiar aquí
        y = get_values(y_vars, (prod, time, scenarios))
        I = get_values(I_vars, (comp, time, scenarios))
        #mostrar_x(x)
        #mostrar_w(w)
        #mostrar_I(I)
        #mostrar_y(y)
        #mostrar_D(D_term, prod, time, scenarios)
    else:
        return print("No se encontro solucion optima.")
    
solver(5,10,50)

# ========================================================================
# INSTANCIA A UTILIZAR (Oh et al.; Slama et al.)
# ========================================================================
# t = 1:10
# epsilon = U[0,25;0,75]
# xi = N[0,2]
# a = 250
# b = 1,5
# L = U[1,3]
# c = U[10,15]
# scenarios = 1000
# ========================================================================