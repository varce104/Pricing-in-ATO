import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import random

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#==============================================================================================================================
# Parámetros: demanda, lead times, BoM, costo compra, costo inventario, precio, probabilidad escenario (unifrome)
#==============================================================================================================================
def demanda_aleatoria(seed, scenarios, prod, time, media=100, desviacion=30, prom=False):
    random.seed(seed)
    demanda = []
    for j in range(prod):
        demanda_p = []
        for t in range(time):
            demanda_t = []
            for s in range(scenarios):
                valor = max(0, int(random.gauss(media, desviacion)))
                demanda_t.append(valor)
            if prom:
                promedio = int(np.mean(demanda_t))
                demanda_p.append(promedio)
            else:
                demanda_p.append(demanda_t)
        demanda.append(demanda_p)
    return demanda

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

def bill_of_materials(comp=15, prod=15, total=10, comparison=True):
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
#==============================================================================================================================

def parametros(comp, prod, time, scenarios, cc=False, ci=False, price=False, prob=False):
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


    if price:
        price_prod = [random.randint(120,160) for j in range(prod)]
    #else:
    #    price_prod = [25,21]
        # ingreso manual


    if prob:
        prob_scenarios = [1/scenarios for s in range(scenarios)] # probabilidad uniforme
        return cost_compra, cost_inv, price_prod, prob_scenarios
    else:
        return cost_compra, cost_inv, price_prod

#==============================================================================================================================
def Recourse_problem(seed, time, scenarios, show=False, xev=None): # Modelo Estocástico (Demanda)
    random.seed(seed)
    A = bill_of_materials()
    comp = len(A)
    prod = len(A[0])
    C, H, P, pi = parametros(comp, prod, time, scenarios, cc=True, ci=True, price=True, prob=True)

    D = demanda_aleatoria(seed, scenarios, prod, time, prom=False)
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
        print("Precios de productos:\n", P)
        print("==============================================================================================")
        #print("")
        print("Demanda:\n", D)
        for s in range(len(D[0][0])):  # recorre escenarios
            print("----------------------------------------------------------------------------------------------")
            print(f"ESCENARIO {s+1}:")
            for j in range(len(D)):
                fila = "  " + "  ".join(f"{D[j][t][s]:>3}" for t in range(len(D[0])))
                print(f"Producto {j+1}:{fila}")
        print("==============================================================================================")
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
    y = m.addVars(prod, time, scenarios, vtype=GRB.INTEGER, name="y")
    I = m.addVars(comp, time, scenarios, vtype=GRB.INTEGER, name="I", lb=0)

    if xev is not None: # Fijar x para modelo EEV
        for i in range(comp):
            for t in range(time):
                if xev.get((i, t)) is not None:
                    x[i, t].LB = float(xev[i, t])
                    x[i, t].UB = float(xev[i, t])

    f = (gp.quicksum(pi[s]*P[j]*y[j,t,s] for s in range(scenarios) for j in range(prod) for t in range(time)) -
    gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
    gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time))
    )

    m.setObjective(f, GRB.MAXIMIZE)

    # restricción de demanda
    m.addConstrs(y[j,t,s] <= D[j][t][s] for j in range(prod) for t in range(time) for s in range(scenarios))

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

    return m, x, y, I, A
#==============================================================================================================================




#==============================================================================================================================
# Cálculo de VSS (Demanda estocástica)
#==============================================================================================================================
def Expected_value(seed, time): # Modelo determinista, sin escenarios
    random.seed(seed)
    A = bill_of_materials()
    comp = len(A)
    prod = len(A[0])
    scenarios = 1
    C, H, P = parametros(comp, prod, time, scenarios, cc=True, ci=True, price=True, prob=False)

    D = demanda_aleatoria(seed,scenarios, prod, time, prom=True)
    L = lead_times(seed, comp, scenarios, time, det=True)

    m = gp.Model("Modelo ATO")

    alpha = {} # alpha[i, tau, t] = 1 si el pedido (i, tau) ya llegó en t
    for i in range(comp):
        for tau in range(time): # Período en que se hace el pedido
            
            # ¡Este es el cambio clave!
            # El lead time depende del período 'tau' en que se pide
            lead_time = L[i][tau] 
            
            for t in range(time): # Período en que se revisa si ya llegó
                if tau + lead_time <= t:
                    alpha[i, tau, t] = 1
                else:
                    alpha[i, tau, t] = 0

    x = m.addVars(comp, time, vtype=GRB.INTEGER, name="x")
    y = m.addVars(prod, time, vtype=GRB.INTEGER, name="y")
    I = m.addVars(comp, time, vtype=GRB.INTEGER, name="I", lb=0)


    f = (gp.quicksum(P[j]*y[j,t] for j in range(prod) for t in range(time)) -
    gp.quicksum(H[i]*I[i,t]for i in range(comp) for t in range(time)) -
    gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(y[j,t] <= D[j][t] for j in range(prod) for t in range(time))

    #m.addConstrs( gp.quicksum(y[j,t-1]*A[i][j] for j in range(prod)) == I[i,t-1] + x_aux(x,i,t,L) - I[i,t] for i in range(comp) for t in range(1,time))
    
    m.addConstrs(
    (gp.quicksum(y[j, tt] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t] ==
     gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
    for i in range(comp) for t in range(time)
    )

    #m.addConstrs(0 <= I[i,t] - gp.quicksum(y[j,t]*A[i][j] for j in range(prod)) for i in range(comp) for t in range(time))
    m.addConstrs(I[i,0] == 0 for i in range(comp))

    m.setParam('OutputFlag', 0)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        x = { (i,t): x[i,t].X for i in range(comp) for t in range(time)}
    else:
        print("No se encontró solución óptima.")

    return x

def EEV(seed,t,s):
    random.seed(seed)
    x_val = Expected_value(seed,t)
    m,x,y,I,A = Recourse_problem(seed,t,s,xev=x_val)
    m.setParam('OutputFlag', 0)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        valor_optimo = m.objVal
        return valor_optimo
    else:
        return None

def VSS(seed,t, s):
    random.seed(seed)
    m,x,y,I,A = Recourse_problem(seed,t, s)
    m.setParam('OutputFlag', 0)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        rp = m.objVal
    if rp == 0:
        return 0
    else:
        eev = EEV(seed,t, s)
        vss = (rp-eev)/rp
        return vss

#==============================================================================================================================
# Cálculo de EVPI (Demanda estocástica)
#==============================================================================================================================

def Wait_and_see(seed, time, s, scenarios):
    random.seed(seed)
    A = bill_of_materials()
    comp = len(A)
    prod = len(A[0])
    C, H, P = parametros(comp, prod, time, scenarios, cc=True, ci=True, price=True, prob=False)

    D = demanda_aleatoria(seed, scenarios, prod, time, prom=False)
    L = lead_times(seed, comp, scenarios, time, det=False)

    alpha = {} # Todos los pedidos que han llegado hasta el período t
    for i in range(comp):
        for l in range(scenarios):
            for tau in range(time): # Período en que se hace el pedido (t)
                lead_time = L[tau][i][l] 
                
                for t in range(time): # Período en que se revisa si ya llegó
                    if tau + lead_time <= t:
                        alpha[i, tau, t, l] = 1
                    else:
                        alpha[i, tau, t, l] = 0

    m = gp.Model("Modelo ATO")

    x = m.addVars(comp, time, vtype=GRB.INTEGER, name="x")
    y = m.addVars(prod, time, vtype=GRB.INTEGER, name="y")
    I = m.addVars(comp, time, vtype=GRB.INTEGER, name="I")

    f = (gp.quicksum(P[j]*y[j,t] for j in range(prod) for t in range(time)) -
    gp.quicksum(H[i]*I[i,t]for i in range(comp) for t in range(time)) -
    gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(y[j,t] <= D[j][t][s] for j in range(prod) for t in range(time))

    m.addConstrs(
    (gp.quicksum(y[j, tt] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t] ==
     gp.quicksum(alpha[i, tau, t,s] * x[i, tau] for tau in range(time)))
    for i in range(comp) for t in range(time)
    )
    #m.addConstrs(0 <= I[i,t] - gp.quicksum(y[j,t]*A[i][j] for j in range(prod)) for i in range(comp) for t in range(time))

    m.addConstrs(I[i,0] == 0 for i in range(comp))

    m.setParam('OutputFlag', 0)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        ws = m.objVal
    return ws

def EVPI(seed, t, scenarios):
    m,x,y,I,A = Recourse_problem(seed,t,scenarios)
    m.setParam('OutputFlag', 0)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        rp = m.objVal
    if rp == 0:
        return 0
    else:
        wsprov = []
        for s in range(scenarios):
            valor = Wait_and_see(seed, t, s, scenarios)
            wsprov.append(valor)

        ws = np.mean(wsprov)
        evpi = (ws - rp)/rp
        return evpi

#==============================================================================================================================
# Mostrar resultados
#==============================================================================================================================


def plot_vss_evpi_boxplots(vss_data, evpi_data):
    """
    Genera boxplots para VSS y EVPI en el mismo gráfico para comparar su dispersión.
    """
    if not vss_data or not evpi_data:
        print("No hay suficientes datos para graficar.")
        return

    data_to_plot = [vss_data, evpi_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    
    bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=['VSS', 'EVPI'])
    colors = ['#D7EAFB', '#FFC3A0']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='red', linewidth=1)
    ax.set_title(f"Dispersión VSS y EVPI ({len(vss_data)} simulaciones)", fontsize=16)
    ax.set_ylabel("Valor (Ratio)", fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Añadir grilla horizontal
    
    print("\nGenerando gráficos de dispersión (boxplots)...")
    plt.show()



#==============================================================================================================================
# Iteración de algoritmo (para demanda incierta)
#==============================================================================================================================
def run_simulations(iter, time, scenarios, max_scenarios=1000):
    vss_results = []
    evpi_results = []
    gap = []

    seed = 42
    for i in range(iter):
        # 1. Calcular VSS
        vss_val = VSS(seed, time, scenarios)
        vss_results.append(vss_val)
        
        # 2. Calcular EVPI
        evpi_val = EVPI(seed, time, scenarios)
        evpi_results.append(evpi_val)
        
        print(f" -> VSS: {vss_val:.4f} | EVPI: {evpi_val:.4f}")
        m,x,y,I,A = Recourse_problem(seed,time,scenarios)
        m.setParam('OutputFlag', 0)
        m.optimize()
        x_val = { (i,t): x[i,t].X for i in range(len(A)) for t in range(time)}
        gap.append({"opt": m.ObjVal, "x_val": x_val})

        seed = seed + 1

    UB = np.mean([g["opt"] for g in gap])

    mejor_resultado = max(gap, key=lambda r: r["opt"])
    mejor_x_val = mejor_resultado["x_val"]

    m,x,y,I,A = Recourse_problem(seed,time,max_scenarios,show=False, xev=mejor_x_val)
    m.setParam('OutputFlag', 0)
    m.optimize()

    LB = m.ObjVal

    gap_opt = (UB - LB)/UB
    media_vss = sum(vss_results)/len(vss_results)
    media_evpi = sum(evpi_results)/len(evpi_results)

    print(f"\nMedia VSS: {media_vss:.4f} | Media EVPI: {media_evpi:.4f}\n")
    print(f"\nGap de Optimalidad: {gap_opt:.4f}\n")

    #plot_vss_evpi_boxplots(vss_results,evpi_results)
        
    return None


#run_simulations(iter=11, time=6, scenarios=20)
#run_simulations(3,10,50)
#run_simulations(21,15,50)
#run_simulations(21,15,200)


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

def mostrar_y(y):
    import pandas as pd
    prod, time, scen = y.shape
    print("VARIABLE y[j,t,s]:")

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
    print("VARIABLE I[i,t,s]:")
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
    
def solver(seed, time,scenarios):
    m, x_vars, y_vars, I_vars, A = Recourse_problem(seed, time, scenarios)
    comp = len(A)
    prod = len(A[0])
    m.setParam('TimeLimit', 1200)
    m.setParam('OutputFlag', 1)
    m.optimize()
    vss_val = VSS(seed, time, scenarios)
    evpi_val = EVPI(seed, time, scenarios)

    if m.status == GRB.OPTIMAL:
        valor_optimo = m.objVal
        print(f"\n Valor optimo: {valor_optimo:.5f}")
        print(f" VSS: {vss_val:.5f}  |  EVPI: {evpi_val:.5f}\n")
        #x = get_values(x_vars, (comp, time))
        #y = get_values(y_vars, (prod, time, scenarios))
        #I = get_values(I_vars, (comp, time, scenarios))
        
        #mostrar_x(x)
        #mostrar_I(I)
        #mostrar_y(y)
    else:
        return print("No se encontró solución óptima.")
    
solver(5,10,50)