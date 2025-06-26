import tkinter as tk
from tkinter.font import Font
from tkinter.messagebox import showerror
from tkinter import ttk
import numpy as np
from scipy.stats import chi2
from random import gauss

def affection_matrice_triangulaire_inferieure(x, L):
    y = np.copy(x)
    y[0] /= L[0, 0]
    for j in range(1, len(x)):
        for k in range(0, j):
            y[j] -= L[j,k]*y[k]
        y[j] /= L[j,j]
    return y

def affection_matrice_matrice_triangulaire_inferieure(X, L):
    Y = np.zeros(np.shape(X))
    for i in range(X.shape[1]):
        Y[:, i] = affection_matrice_triangulaire_inferieure(X[:, i], L)
    return Y

def affection_matrice_triangulaire_superieure(x, L):
    y = np.copy(x)
    m = len(x)-1
    y[m] /= L[m, m]
    for j in range(m-1, -1, -1):
        for k in range(j+1, m+1):
            y[j] -= L[j, k]*y[k]
        y[j] /= L[j,j]
    return y

def comparaison(Chi_2:float, df:int, p:float=0.95):
    if Chi_2 < chi2.ppf(p, df):
        return True
    else:
        return False

def funct_OLS(x, y):
    # ISO TS 28037-2010 : 5.8 minimization of sum(y - Ax - b)^2
    # Equation : y = ax + b

    # transform to numpy array
    x, y = np.array(x, dtype='float'), np.array(y, dtype='float')
    # Step 1
    x0, y0 = np.mean(x), np.mean(y)
    # Step 2
    x_, y_ = x-x0, y-y0
    # Step 3
    a = np.sum(x_*y_)/np.sum(np.power(x_, 2))
    b = y0 - a*x0

    # Step 4
    ua_2 = 1/np.sum(np.power(x_, 2))
    ub_2 = 1/len(x) + x0**2/np.sum(np.power(x_, 2))
    cov_a_b = -x0**2/np.sum(np.power(x_, 2))

    # validation of the model
    if len(x)>2:
        r = (y-a*x-b)
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1
    return a, b, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2

def funct_WLS(x, y, u_y):
    # Méthode des moindres carrés pondérés
    # ISO TS 28037-2010 : 6 minimization of sum(y - Ax - b)²w²
    # Equation : y/u_y = ax/u_y + b/u_y
    # transform to numpy array
    x, y, w = np.array(x, dtype='float'), np.array(y, dtype='float'), 1/(1.e-15+np.array(u_y, dtype='float'))

    # Step 1
    F2 = np.sum(np.power(w, 2))
    # Step 2
    g0 = np.sum(np.power(w, 2)*x)/F2
    h0 = np.sum(np.power(w, 2)*y)/F2
    # Step 3
    g = w*(x-g0)
    h = w*(y-h0)
    # Step 4
    G2 = np.sum(np.power(g, 2))
    # Step 5
    a = np.sum(g*h)/G2
    b = h0 - a*g0
    # Step 6
    ub_2 = 1/F2 + np.power(g0, 2)/G2
    ua_2 = 1/G2
    cov_a_b = -g0/G2

    # 6.3 validation of the model
    if len(x)>2:
        r = w*(y-a*x-b)
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1
    return a, b, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2

def func_GDR(x, y, u_x, u_y, eps:float=0.00005):
    # Méthode des moindres carrés avec des erreurs sur les variables 
    # ISO TS 28037-2010 : 7 minimization of sum[v²(x-X)& + w²(y - Ax - b)²]
    # Equation : y/u_y = ax/u_y + b/u_y
    # transform to numpy array
    x, y = np.array(x, dtype='float'), np.array(y, dtype='float')
    u_x, u_y = np.array(u_x, dtype='float'), np.array(u_y, dtype='float')

    # Step 1
    a_, b_, _, __, ___, ____ = funct_WLS(x, y, u_y)
    delta_a, delta_b = 10*eps, 10*eps
    while abs(delta_a) > eps and abs(delta_b) > eps: 
        # Step 2
        t = 1/(np.power(u_y, 2) + a_**2 * np.power(u_x, 2) + 1.e-15)
        x_ = (x*np.power(u_y, 2) + (y-b_)*a_*np.power(u_x, 2))*t
        z = y - a_*x - b_

        # Step 3
        f = np.power(t, 1/2)
        g = f*x_
        h = f*z

        # Step 4
        F2 = np.sum(t)
        g0 = np.sum(f*g)/F2
        h0 = np.sum(f*h)/F2
        g_ = g - f*g0
        h_ = h - f*h0
        G2_ = np.sum(np.power(g_, 2))

        delta_a = np.sum(g_*h_)/G2_
        delta_b = h0 - delta_a*g0

        # Step 5
        a_ += delta_a
        b_ += delta_b
    r = h_ - delta_b*g_

    # Step 7
    ua_2 = 1/ G2_
    ub_2 = 1/F2 + g0**2/G2_
    cov_a_b = -g0/G2_

    # 6.3 validation of the model
    if len(x)>2:
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1
    return a_, b_, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2

def func_GDR_cov(x, y, u_x, u_y, cov, eps:float=0.00005):
    # Méthode des moindres carrés avec des erreurs corrélées sur les variables et des covariances
    # ISO TS 28037-2010 : 8  
    # transform to numpy array
    x, y = np.array(x, dtype='float'), np.array(y, dtype='float')
    u_x, u_y, cov = np.array(u_x, dtype='float'), np.array(u_y, dtype='float'), np.array(cov, dtype='float')

    # Step 1
    a_, b_, _, __, ___, ____ = funct_WLS(x, y, u_y)
    delta_a, delta_b = 10*eps, 10*eps
    while abs(delta_a) > eps and abs(delta_b) > eps: 
        # Step 2
        t = 1/(np.power(u_y, 2) - 2*a_*cov + a_**2 * np.power(u_x, 2) + 1.e-15)
        x_ = (x*(np.power(u_y, 2)-a_*cov) - (y-b_)*(cov-a_*np.power(u_x, 2)))*t
        z = y - a_*x - b_

        # Step 3
        f = np.power(t, 1/2)
        g = f*x_
        h = f*z

        # Step 4
        F2 = np.sum(t)
        g0 = np.sum(f*g)/F2
        h0 = np.sum(f*h)/F2
        g_ = g - f*g0
        h_ = h - f*h0
        G2_ = np.sum(np.power(g_, 2))

        delta_a = np.sum(g_*h_)/G2_
        delta_b = h0 - delta_a*g0

        # Step 5
        a_ += delta_a
        b_ += delta_b
    r = h_ - delta_b*g_

    # Step 7
    ua_2 = 1/ G2_
    ub_2 = 1/F2 + g0**2/G2_
    cov_a_b = -g0/G2_

    # 6.3 validation of the model
    if len(x)>2:
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1
    return a_, b_, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2

def func_GLS(x, y, cov_y):
    # Méthode des moindres carrés généralisées avec des erreurs corrélées sur les y
    # # ISO TS 28037-2010 : 9
    # transform to numpy array
    x, y, cov_y = np.array(x, dtype='float'), np.array(y, dtype='float'), np.array(cov_y, dtype='float') 
    # Step 1
    L_y = np.linalg.cholesky(cov_y)

    # Step 2 
    f = affection_matrice_triangulaire_inferieure(np.ones(len(x)), L_y)
    g = affection_matrice_triangulaire_inferieure(x, L_y)
    h = affection_matrice_triangulaire_inferieure(y, L_y)

    # Step 3
    F2 = np.sum(np.power(f, 2))

    # Step 4
    g0 = np.sum(f*g)/F2
    h0 = np.sum(f*h)/F2

    # Step 5
    g_ = g- g0*f
    h_ = h - h0*f

    # Step 6
    G2_ = np.sum(np.power(g_, 2))

    # Step 7
    a = np.sum(g_*h_)/G2_
    b = h0 - a*g0

    # Step 8
    ub_2 = 1/F2 + np.power(g0, 2)/G2_
    ua_2 = 1/G2_
    cov_a_b = -g0/G2_

    # 9.3 validation of the model
    if len(x)>2:
        r = h_ - a*g_
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1
    return a, b, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2

def func_GGMR(x, y, cov_xy, eps:float=0.00005):
    # Méthode des moindres carrés ordinaires
    # # ISO TS 28037-2010 : 10
    # transform to numpy array
    x, y, cov_xy = np.array(x, dtype='float'), np.array(y, dtype='float'), np.array(cov_xy, dtype='float') 
    
    # Step 1
    u_y = np.array([np.sqrt(cov_xy[i, i]) for i in range(len(x), cov_xy.shape[0])], dtype=float)
    a_, b_, _, __, ___, ____ = funct_WLS(x, y, u_y)

    t_ = np.array(list(x) + [b_, a_])
    delta_a = 10*eps
    delta_b = 10*eps

    # Step 3
    L = np.linalg.cholesky(cov_xy)
    
    while abs(delta_a) > eps and abs(delta_b) > eps:
        a_ = t_[len(t_)-1]
        b_ = t_[len(t_)-2]
        x_ = t_[:len(t_)-2]

        # Step 2
        f = np.concatenate((x-x_, y-(a_*x_+b_)))
        J = np.zeros((2*len(x), len(x)+2), dtype=float)
        J[:len(x), :len(x)] = np.diag(-np.ones(len(x)))
        J[len(x):, :len(x)] = np.diag(-a_*np.ones(len(x)))
        J[len(x):, len(x)] = -np.ones(len(x))
        J[len(x):, len(x)+1] = -x_

        # Step 4
        f_ = affection_matrice_triangulaire_inferieure(f, L)
        J_ = affection_matrice_matrice_triangulaire_inferieure(J, L)

        # Step 5
        g = J_.T@f_
        H = J_.T@J_

        # Step 6
        M = np.linalg.cholesky(H)

        # Step 7
        q = affection_matrice_triangulaire_inferieure(-g, M)

        # Step 8 
        delta_t = affection_matrice_triangulaire_superieure(q, M.T)

        # Step 9
        t_ += delta_t
        delta_a = delta_t[len(t_)-1]
        delta_b = delta_t[len(t_)-2]
    
    # Step 10
    a = t_[len(t_)-1]
    b = t_[len(t_)-2]

    # Step 11
    M_22 = M[-2:, -2:]

    ub_2 = (M_22[1, 1]**2 + M_22[1, 0]**2)/(M_22[0, 0]**2 * M_22[1, 1]**2)
    ua_2 = 1/M_22[1, 1]**2
    cov_a_b = - M_22[1, 0]/(M_22[0, 0]*(M_22[1, 1,]**2))
    
    # 10.3 validation of the model
    if len(x)>2:
        Chi_2 = np.sum(np.power(f_, 2))
    else:
        Chi_2 = -1
    return a, b, np.sqrt(ua_2), np.sqrt(ub_2), cov_a_b, Chi_2
                                              

class EntryPopup(ttk.Entry):
    def __init__(self, tree, iid, column, text, **kw):
        super().__init__(tree, **kw)
        self.tv = tree  # reference to parent window's treeview
        self.iid = iid  # row id
        self.column = column 

        self.insert(0, text) 
        self['exportselection'] = False  # Prevents selected text from being copied to  
                                         # clipboard when widget loses focus
        self.focus_force()  # Set focus to the Entry widget
        self.select_all()   # Highlight all text within the entry widget
        self.bind("<Return>", self.on_return) # Enter key bind
        self.bind("<Control-a>", self.select_all) # CTRL + A key bind
        self.bind("<Escape>", lambda *ignore: self.destroy()) # ESC key bind
        
    def on_return(self, event):
        '''Insert text into treeview, and delete the entry popup'''
        rowid = self.tv.focus()  # Find row id of the cell which was clicked
        vals = self.tv.item(rowid, 'values')  # Returns a tuple of all values from the row with id, "rowid"
        vals = list(vals)  # Convert the values to a list so it becomes mutable
        vals[self.column] = self.get()  # Update values with the new text from the entry widget
        self.tv.item(rowid, values=vals)  # Update the Treeview cell with updated row values
        self.destroy()  # Destroy the Entry Widget
        
    def sauv(self):
        '''Insert text into treeview, and delete the entry popup'''
        rowid = self.tv.focus()  # Find row id of the cell which was clicked
        vals = self.tv.item(rowid, 'values')  # Returns a tuple of all values from the row with id, "rowid"
        vals = list(vals)  # Convert the values to a list so it becomes mutable
        vals[self.column] = self.get()  # Update values with the new text from the entry widget
        self.tv.item(rowid, values=vals)  # Update the Treeview cell with updated row values
        self.destroy()  # Destroy the Entry Widget

    def select_all(self, *ignore):
        ''' Set selection on the whole text '''
        self.selection_range(0, 'end')
        return 'break' # returns 'break' to interrupt default key-bindings

class app:
    def __init__(self):
        # Variables
        self._frame = None
        self._L_donnee = []
        
        self._fn = tk.Tk()
        self._fn.geometry("530x490")
        self._fn.resizable(0, 0) #Don't allow resizing in the x or y direction
        self._police =  Font(family = "Time", size = 10)
        self._police_g =  Font(family = "Time", size = 13)
        self._fn.title('Calibration')

        for i in range(3):
            self._fn.rowconfigure(i, weight=1)
            self._fn.columnconfigure(i, weight=1)

        self._s = ttk.Style()
        self._s.theme_use('winnative')

        self._parametre()
        self._affichage_variable()
        self._cadre_choix_modele()

        self._fn.mainloop()

    def _parametre(self):
        self._var_nb_mes = tk.IntVar(self._fn, value =5)

        cdr = ttk.Frame(self._fn)
        cdr.place(x=185, y = 0, height=50, width=160)

        lb = ttk.Label(cdr, text = "Number of measures: ")
        lb.place(x=0, y = 0, height=20, width=120) 

        sp = ttk.Spinbox(cdr, from_=2, to=99, width=2,
                         textvariable = self._var_nb_mes)
        sp.place(x=120, y = 0, height=20, width=80)
        
        bout = ttk.Button(cdr, text="Validate",
                          command=self._affichage_variable)
        bout.place(x=60, y = 20, height=25, width=60)

    def _affichage_variable(self):
        dx = 41
        dy = 40
        if int(self._var_nb_mes.get()) < 2 :
            self._var_nb_mes.set(2)
        elif int(self._var_nb_mes.get()) > 99 :
            self._var_nb_mes.set(99)

        if self._var_nb_mes.get()<5:
            nb_dim = self._var_nb_mes.get()
        else:
            nb_dim = 5

        if self._frame != None:
            self._tree_XY.destroy()
            self._frame.destroy()

        off_y = 50

        L_dim = [125, 125, 125, 125] # 2, 3, 4, 5
        L_dim_cov = [242, 242, 242, 242]
        label = tk.LabelFrame(self._frame, text="Value: ", font=self._police_g)
        if nb_dim <5 :
            label.place(x=61, y = off_y, height=L_dim[nb_dim-2]+dy, width=3*dx+dx//2)
        else:
            label.place(x=61, y = off_y, height=L_dim[nb_dim-2]+dy, width=3.5*dx)
        
        if self._var_nb_mes.get()<5:
            self._tree_XY = ttk.Treeview(label, columns=("Nom", "X", "Y"), show="headings", height=self._var_nb_mes.get())
        else:
            self._tree_XY = ttk.Treeview(label, columns=("Nom", "X", "Y"), show="headings", height=5)

            scrollbar = ttk.Scrollbar(label, orient="vertical", command=self._tree_XY.yview)
            self._tree_XY.configure(yscrollcommand=scrollbar.set)
            scrollbar.place(x=3*dx, y=0, height=L_dim[nb_dim-2])

        self._tree_XY.bind("<Double-1>", lambda event: self._onDoubleClick(event))

        self._tree_XY.heading("Nom", text="N°")
        self._tree_XY.heading("X", text="X")
        self._tree_XY.heading("Y", text="Y")

        self._tree_XY.column("#0", anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_XY.column("#1", anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_XY.column("#2", anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_XY.column("#3", anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        for i in range(self._var_nb_mes.get()):
            self._tree_XY.insert("", "end", values=(i+1, 0,0))
        
        self._tree_XY.place(height=L_dim[nb_dim-2], width=3*dx)

        # Cov x
        label = tk.LabelFrame(self._frame, text="Covariance X: ", font=self._police_g)
        if nb_dim <5 :
            label.place(x=0, y = off_y+L_dim[nb_dim-2]+dy, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)
        else:
            label.place(x=0, y = off_y+L_dim[nb_dim-2]+dy, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)

        L_nom_X = [""]+["X" + str(i+1) for i in range(self._var_nb_mes.get())]
        if self._var_nb_mes.get()<5:
            self._tree_cov_X = ttk.Treeview(label, columns=tuple(L_nom_X), show="headings", height=self._var_nb_mes.get())
        else:
            self._tree_cov_X = ttk.Treeview(label, columns=tuple(L_nom_X), show="headings", height=5)

            scrollbar = ttk.Scrollbar(label, orient="vertical", command=self._tree_cov_X.yview)
            self._tree_cov_X.configure(yscrollcommand=scrollbar.set)
            scrollbar.place(x=L_dim_cov[nb_dim-2], y=0, height=L_dim[nb_dim-2])

            scrollbar = ttk.Scrollbar(label, orient="horizontal", command=self._tree_cov_X.xview)
            self._tree_cov_X.configure(xscrollcommand=scrollbar.set)
            scrollbar.place(y=L_dim[nb_dim-2], x=0, width=L_dim_cov[nb_dim-2])

        self._tree_cov_X.bind("<Double-1>", lambda event: self._onDoubleClick(event))

        for i in range(len(L_nom_X)):
            self._tree_cov_X.heading(L_nom_X[i], text=L_nom_X[i])
            self._tree_cov_X.column("#"+str(i), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_cov_X.column("#"+str(len(L_nom_X)), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)

        L_val = [0 for i in range(self._var_nb_mes.get())]
        for i in range(self._var_nb_mes.get()):
            self._tree_cov_X.insert("", "end", values=tuple(["X"+str(i+1)]+L_val))
    

        self._tree_cov_X.place(x=0, y=0, height=L_dim[nb_dim-2], width=L_dim_cov[nb_dim-2])

        # Cov y
        label = tk.LabelFrame(self._frame, text="Covariance Y: ", font=self._police_g)
        if nb_dim <5 :
            label.place(x=L_dim_cov[nb_dim-2]+dx//2, y = off_y, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)
        else:
            label.place(x=L_dim_cov[nb_dim-2]+dx//2, y = off_y, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)

        L_nom_Y = [""]+["Y" + str(i+1) for i in range(self._var_nb_mes.get())]
        if self._var_nb_mes.get()<5:
            self._tree_cov_Y = ttk.Treeview(label, columns=tuple(L_nom_Y), show="headings", height=self._var_nb_mes.get())
        else:
            self._tree_cov_Y = ttk.Treeview(label, columns=tuple(L_nom_Y), show="headings", height=5)

            scrollbar = ttk.Scrollbar(label, orient="vertical", command=self._tree_cov_Y.yview)
            self._tree_cov_Y.configure(yscrollcommand=scrollbar.set)
            scrollbar.place(x=L_dim_cov[nb_dim-2], y=0, height=L_dim[nb_dim-2])

            scrollbar = ttk.Scrollbar(label, orient="horizontal", command=self._tree_cov_Y.xview)
            self._tree_cov_Y.configure(xscrollcommand=scrollbar.set)
            scrollbar.place(y=L_dim[nb_dim-2], x=0, width=L_dim_cov[nb_dim-2])

        self._tree_cov_Y.bind("<Double-1>", lambda event: self._onDoubleClick(event))

        for i in range(len(L_nom_Y)):
            self._tree_cov_Y.heading(L_nom_Y[i], text=L_nom_Y[i])
            self._tree_cov_Y.column("#"+str(i), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_cov_Y.column("#"+str(len(L_nom_Y)), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)

        L_val = [0 for i in range(self._var_nb_mes.get())]
        for i in range(self._var_nb_mes.get()):
            self._tree_cov_Y.insert("", "end", values=tuple(["Y"+str(i+1)]+L_val))
    
        self._tree_cov_Y.place(x=0, y=0, height=L_dim[nb_dim-2], width=L_dim_cov[nb_dim-2])

        # Cov XY
        label = tk.LabelFrame(self._frame, text="Covariance XY: ", font=self._police_g)
        if nb_dim <5 :
            label.place(x=L_dim_cov[nb_dim-2]+dx//2, y = off_y+L_dim[nb_dim-2]+dy, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)
        else:
            label.place(x=L_dim_cov[nb_dim-2]+dx//2, y = off_y+L_dim[nb_dim-2]+dy, height=L_dim[nb_dim-2]+dy, width=L_dim_cov[nb_dim-2]+dx//2)

        L_nom_Y = [""]+["Y" + str(i+1) for i in range(self._var_nb_mes.get())]
        if self._var_nb_mes.get()<5:
            self._tree_cov_XY = ttk.Treeview(label, columns=tuple(L_nom_Y), show="headings", height=self._var_nb_mes.get())
        else:
            self._tree_cov_XY = ttk.Treeview(label, columns=tuple(L_nom_Y), show="headings", height=5)

            scrollbar = ttk.Scrollbar(label, orient="vertical", command=self._tree_cov_XY.yview)
            self._tree_cov_XY.configure(yscrollcommand=scrollbar.set)
            scrollbar.place(x=L_dim_cov[nb_dim-2], y=0, height=L_dim[nb_dim-2])

            scrollbar = ttk.Scrollbar(label, orient="horizontal", command=self._tree_cov_XY.xview)
            self._tree_cov_XY.configure(xscrollcommand=scrollbar.set)
            scrollbar.place(y=L_dim[nb_dim-2], x=0, width=L_dim_cov[nb_dim-2])

        self._tree_cov_XY.bind("<Double-1>", lambda event: self._onDoubleClick(event))

        for i in range(len(L_nom_Y)):
            self._tree_cov_XY.heading(L_nom_Y[i], text=L_nom_Y[i])
            self._tree_cov_XY.column("#"+str(i+1), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)
        self._tree_cov_XY.column("#"+str(len(L_nom_Y)), anchor=tk.CENTER, stretch=0, width=40, minwidth=40)

        L_val = [0 for i in range(self._var_nb_mes.get())]
        for i in range(self._var_nb_mes.get()):
            self._tree_cov_XY.insert("", "end", values=tuple(["X"+str(i+1)]+L_val))
    
        self._tree_cov_XY.place(x=0, y=0, height=L_dim[nb_dim-2], width=L_dim_cov[nb_dim-2])

        self._dim_y = 2*(L_dim[nb_dim-2]+dy)+off_y

    def _onDoubleClick(self, event):
            '''Executed, when a row is double-clicked'''
            # close previous popups
            try:  # in case there was no previous popup
                self._entryPopup.destroy()
            except AttributeError:
                pass

            tree = event.widget

            # what row and column was clicked on
            rowid = tree.identify_row(event.y)
            column = tree.identify_column(event.x)

            # return if the header was double clicked
            if not rowid:
                return

            if column == "#1":
                return
            # get cell position and cell dimensions
            x, y, width, height = tree.bbox(rowid, column)

            # y-axis offset
            pady = height // 2

            # place Entry Widget
            text = tree.item(rowid, 'values')[int(column[1:])-1]
            self._entryPopup = EntryPopup(tree, rowid, int(column[1:])-1, text)
            self._entryPopup.place(x=x, y=y+pady, width=width, height=height, anchor='w')

    def _cadre_choix_modele(self):
        off_y = 380

        lb = ttk.LabelFrame(self._fn, text = "Calibration model:")
        lb.place(x=0, y=off_y, height=50, width = 523)

        self._L_nom_modele = ["Without uncertainties",
                        "For uncertainties associated with the yi: Least Squares",
                        "For uncertainties associated with the yi: Monte-Carlo",
                        "For uncertainties associated with the xi and the yi: Least Squares",
                        "For uncertainties associated with the xi and the yi: Monte-Carlo",
                        "For uncertainties associated with the xi and the yi and covariances associated with (xi, yi)",
                        "For uncertainties and covariances associated with the yi",
                        "For uncertainties and covariances associated with the xi and the yi",]
        self._L_combo = ttk.Combobox(lb, values=self._L_nom_modele)
        self._L_combo.current(0)

        self._L_combo.place(x=0, y=0, height=25, width = 518)

        lab = ttk.Label(self._fn, text="Precision:")
        lab.place(x=88, y = off_y+67, height=25, width=60)

        self._var_pre = tk.IntVar(self._fn, value=2)
        lab = ttk.Entry(self._fn, width=12,
                         textvariable = self._var_pre)
        lab.place(x=88+60, y = off_y+67, height=25, width=30)


        bout = ttk.Button(self._fn, text="Calculate",
                          command=self._calcul)
        bout.place(x=103+265, y = off_y+67, height=25, width=60)

    def _calcul(self):
        if self._L_nom_modele[0] == self._L_combo.get():
            L_X = []
            L_Y = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                a, b, ua, ub, cov_a_b, chi_2 = funct_OLS(L_X, L_Y)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[1] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_uY = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))
                compt = 0

                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_uY.append(float(L[compt])**(1/2))

                a, b, ua, ub, cov_a_b, chi_2 = funct_WLS(L_X, L_Y, L_uY)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[2] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_uY = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))
                
                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_uY.append(float(L[compt])**(1/2))

                a, b, ua, ub, cov_a_b, chi_2 = Monte_carlo(L_X, L_Y, 
                                                           [0 for elmt in L_X], L_uY, 
                                                           self._var_pre.get())
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[3] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_uY = []
            L_uX = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_X.item(parent)["values"]
                    L_uX.append(float(L[compt])**(1/2))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_uY.append(float(L[compt])**(1/2))
                a, b, ua, ub, cov_a_b, chi_2 = func_GDR(L_X, L_Y, L_uX, L_uY)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[4] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_uY = []
            L_uX = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_X.item(parent)["values"]
                    L_uX.append(float(L[compt])**(1/2))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_uY.append(float(L[compt])**(1/2))
                a, b, ua, ub, cov_a_b, chi_2 = Monte_carlo(L_X, L_Y, 
                                                           L_uX, L_uY, 
                                                           self._var_pre.get())
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[5] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_uY = []
            L_uX = []
            L_uXY = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_X.item(parent)["values"]
                    L_uX.append(float(L[compt])**(1/2))

                compt = 0
                for parent in self._tree_cov_Y.get_children():
                    compt += 1
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_uY.append(float(L[compt])**(1/2))

                for parent in self._tree_cov_XY.get_children():
                    L = self._tree_cov_XY.item(parent)["values"]
                    L_uXY.append([float(L[i]) for i in range(1, len(L))]) 
                a, b, ua, ub, cov_a_b, chi_2 = func_GDR_cov(L_X, L_Y, L_uX, L_uY, L_uXY)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        elif self._L_nom_modele[6] == self._L_combo.get():
            L_X = []
            L_Y = []
            L_cov_y = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                for parent in self._tree_cov_Y.get_children():
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_cov_y.append([float(L[i]) for i in range(1, len(L))]) 
                a, b, ua, ub, cov_a_b, chi_2 = func_GLS(L_X, L_Y, L_cov_y)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

        else:
            L_X = []
            L_Y = []
            L_cov_x = []
            L_cov_y = []
            L_cov_xy = []
            try:
                for parent in self._tree_XY.get_children():
                    L = self._tree_XY.item(parent)["values"]
                    L_X.append(float(L[1]))
                    L_Y.append(float(L[2]))

                for parent in self._tree_cov_X.get_children():
                    L = self._tree_cov_X.item(parent)["values"]
                    L_cov_x.append([float(L[i]) for i in range(1, len(L))]) 

                for parent in self._tree_cov_Y.get_children():
                    L = self._tree_cov_Y.item(parent)["values"]
                    L_cov_y.append([float(L[i]) for i in range(1, len(L))]) 

                for parent in self._tree_cov_XY.get_children():
                    L = self._tree_cov_XY.item(parent)["values"]
                    L_cov_xy.append([float(L[i]) for i in range(1, len(L))]) 
                L_cov = np.concatenate((np.concatenate((L_cov_x, L_cov_xy), axis=1), 
                                        np.concatenate((L_cov_xy, L_cov_y), axis=1)), axis=0)
                a, b, ua, ub, cov_a_b, chi_2 = func_GGMR(L_X, L_Y, L_cov)
                self._fn_second(a, b, ua, ub, cov_a_b, chi_2, len(L_X))
            except:
                showerror(title="Error", message="Data problem")

    def _fn_second(self, a, b, ua, ub, cov_a_b, chi_2, dim):
            fn = tk.Toplevel()
            fn.title("Result")
            text = "y =ax+b"
            ttk.Label(fn, text = text).grid(row=0, column=0)                
            text = "a±1σ: "+str(round(a, self._var_pre.get()))+"±"+str(round(ua, self._var_pre.get()))
            ttk.Label(fn, text = text).grid(row=1, column=0)
            text = "b±1σ: "+str(round(b, self._var_pre.get()))+"±"+str(round(ub, self._var_pre.get()))
            ttk.Label(fn, text = text).grid(row=2, column=0)
            text = "Cov(a,b): "+str(round(cov_a_b, self._var_pre.get()))
            ttk.Label(fn, text = text).grid(row=3, column=0)
            text = "χ obs: "+str(round(chi_2, self._var_pre.get()))
            ttk.Label(fn, text = text).grid(row=4, column=0)
            text = "Degree of freedom: "+str(dim-2)
            ttk.Label(fn, text = text).grid(row=5, column=0)
            text = "Valid model: "+str(comparaison(chi_2, dim-2))
            ttk.Label(fn, text = text).grid(row=6, column=0)
            fn.mainloop()        

def Monte_carlo(L_x:list, L_y:list, L_s_x:list, L_s_y:list, pre:int):
    L_a = []
    L_b = []
    for i in range(10**(2*pre)):
        X = [L_x[i] + gauss(0,L_s_x[i]) for i in range(len(L_x))]
        Y = [L_y[i] + gauss(0,L_s_y[i]) for i in range(len(L_y))]
        a, b, _, __, ___, ____ = funct_OLS(X, Y)
        L_a.append(a)
        L_b.append(b)

    a = np.mean(L_a)
    b = np.mean(L_b)

    ua = np.std(L_a)
    ub = np.std(L_b)

    cov_a_b = 1/len(L_x) - ub**2

    if len(L_x)>2:
        r = (np.array(L_y)-a*np.array(L_x)-b)
        Chi_2 = np.sum(np.power(r, 2))
    else:
        Chi_2 = -1

    return a, b, ua, ub, cov_a_b, Chi_2

if '__main__' == __name__:
    fn = app()
