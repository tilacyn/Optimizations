import numpy as np
import pandas as pd
import random

class SimplexMethod:
    def __init__(self):
        pass

    def solve(self, A, b, c, task='min', ineq='more', game=False):
        if task != 'min' and task != 'max':
            raise KeyError('Task should be \'min\' or \'max\', %r was passed' % task)
        if ineq != 'more' and ineq != 'less':
            raise KeyError('Ineq should be \'more\' or \'less\', %r was passed' % ineq)
        if ineq == 'more':
            b = -b
            A = -A
        table = self.find_basic(A, b, c)
        print('Table with basic plan:')
        print(table)
        coef_mult = 1
        if task == 'max':
            coef_mult = -1
        while (table.loc['Q'].drop(labels=['1']) * coef_mult > 0).any():
            func_row = table.loc['Q'].drop(labels=['1']) * coef_mult
            idx = func_row[func_row > 0].index
            column = idx[random.choice(range(len(idx)))]
            row = self.choose_row(table, column)
            table = self.jordan_step(table, row, column)
        print('Resulting table:')
        print(table)
        print('Function extremum:', table.loc['Q', '1'])
        print('Optimal plan:')
        plan = pd.Series(np.zeros(A.shape[1] + A.shape[0]))
        plan.index = ['x' + str(int(i + 1)) for i in range(A.shape[1] + A.shape[0])]
        plan[table[:-1]['1'].index] = table[:-1]['1']
        plan = plan.drop(labels=['x' + str(int(i + 1)) for i in range(A.shape[1], A.shape[1] + A.shape[0], 1)])
        if game:
            plan /= np.sum(plan)
        print(plan)
        if game:
            print('Optimal plan for another player:')
            other_plan = pd.Series(np.zeros(A.shape[1] + A.shape[0]))
            other_plan.index = ['-x' + str(int(i + 1)) for i in range(A.shape[1] + A.shape[0])]
            other_plan[table.loc['Q'].drop(labels=['1']).index] = -coef_mult * table.loc['Q'].drop(labels=['1'])
            other_plan = other_plan.drop(labels=['-x' + str(int(i + 1)) for i in range(A.shape[1])])
            other_plan.index = ['x' + str(int(i + 1 - A.shape[1])) for i in range(A.shape[1], A.shape[1] + A.shape[0], 1)]
            other_plan /= np.sum(other_plan)
            print(other_plan)
        return table.loc['Q', '1'], plan

    def find_basic(self, A, b, c):
        table = self.init_table(A, b, c)
        print('Initialized table:')
        print(table)
        while not (table[:-1]['1'] > 0).all():
            idx = table[:-1]['1'][table[:-1]['1'] < 0].index
            base_row = idx[random.choice(range(len(idx)))]
            if not (table.loc[base_row].drop(labels=['1']) < 0).any():
                raise ValueError('Row %r has no negative elements!' % base_row)
            base_row_values = table.loc[base_row].drop(labels=['1'])
            idx = base_row_values[base_row_values < 0].index
            column = idx[random.choice(range(len(idx)))]
            row = self.choose_row(table, column)
            table = self.jordan_step(table, row, column)
        return table

    def choose_row(self, table, column):
        column_divided = table[:-1]['1'] / table[:-1][column]
        return column_divided.where(column_divided > 0).idxmin()

    def init_table(self, A, b, c):
        table = pd.DataFrame(A)
        table = table.append(-pd.Series(c), ignore_index=True)
        table.columns = ['-x' + str(i + 1) for i in range(A.shape[1])]
        table['1'] = np.append(b, 0)
        table.index = ['x' + str(int(i + 1)) for i in range(A.shape[1], A.shape[1] + A.shape[0], 1)] + ['Q']
        return table

    def jordan_step(self, table, x, y):
        elem = table.loc[x, y]
        row = table.loc[x].copy()
        row[y] = 0
        column = table[y].copy()
        column[x] = 0
        table[y] /= -elem
        table.loc[x] /= elem
        table.loc[x, y] = 1 / elem
        for col in table.columns:
            table[col] -= column * row[col] / elem
        table = table.rename(index={x : y[1:]}, columns={y: '-' + x})
        return table

def f(x, y):
    if x < y:
        return 1
    if x == y:
        return 0
    return -1

A = np.array([[(6 * min(x, y) - (6 - min(x, y)) * max(x, y)) * f(x, y) + 24 for y in range(1,7,1)] for x in range(1,7,1)])
#A = np.array([[2, 1, 4], [0, 3, 2], [9, 6, 1], [6, 0, 3]])
#A = np.array([[6, 7, 4, 1], [11, 6, 13, 11], [11, 1, 6, 21]])
b = np.ones(6)
c = np.ones(6)

#первый игрок - A.T, min, more; второй - A, max, less

s = SimplexMethod()
val, plan = s.solve(A, b, c, task='max', ineq='less', game=True)