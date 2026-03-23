import tkinter as tk
from tkinter import ttk, scrolledtext
import random
import math
import statistics
import time
import threading

GRID_SIZE=32
CELL_SIZE=18
CANVAS_SIZE=GRID_SIZE*CELL_SIZE
TRAP_LIFETIME_MIN=3
TRAP_LIFETIME_MAX=8
TRAP_SPAWN_PROB=0.12
MAX_STEPS_PER_RUN=300
NUM_STATES=8
NUM_INPUTS=16
ACTION_FORWARD=0
ACTION_LEFT=1
ACTION_RIGHT=2
ACTION_STAY=3
ACTIONS=["FORWARD","LEFT","RIGHT","STAY"]
DIR_UP=0
DIR_RIGHT=1
DIR_DOWN=2
DIR_LEFT=3
DIRS={DIR_UP:(0,-1),DIR_RIGHT:(1,0),DIR_DOWN:(0,1),DIR_LEFT:(-1,0)}

def clamp(x,a,b):
    return max(a,min(b,x))

def wrap_pos(x,y):
    return x%GRID_SIZE,y%GRID_SIZE

class GameEnvironment:
    def __init__(self,seed=None):
        self.random=random.Random(seed)
        self.reset()
    def reset(self):
        self.agent_x=GRID_SIZE//2
        self.agent_y=GRID_SIZE//2
        self.agent_dir=DIR_UP
        self.alive=True
        self.steps=0
        self.stay_count=0
        self.traps={}
    def cell_has_trap(self,x,y):
        x,y=wrap_pos(x,y)
        return (x,y)in self.traps
    def update_traps(self):
        to_delete=[]
        for pos in self.traps:
            self.traps[pos]-=1
            if self.traps[pos]<=0:
                to_delete.append(pos)
        for pos in to_delete:
            del self.traps[pos]
        if self.random.random()<TRAP_SPAWN_PROB:
            count_new=self.random.randint(1,3)
            for _ in range(count_new):
                x=self.random.randint(0,GRID_SIZE-1)
                y=self.random.randint(0,GRID_SIZE-1)
                if (x,y)==(self.agent_x,self.agent_y):
                    continue
                ttl=self.random.randint(TRAP_LIFETIME_MIN,TRAP_LIFETIME_MAX)
                self.traps[(x,y)]=ttl
    def get_relative_cell(self,rel_dir):
        real_dir=(self.agent_dir+rel_dir)%4
        dx,dy=DIRS[real_dir]
        x,y=wrap_pos(self.agent_x+dx,self.agent_y+dy)
        return x,y
    def get_input_code(self):
        fx,fy=self.get_relative_cell(0)
        lx,ly=self.get_relative_cell(-1)
        rx,ry=self.get_relative_cell(1)
        cx,cy=self.agent_x,self.agent_y
        bits=[1 if self.cell_has_trap(fx,fy)else 0,1 if self.cell_has_trap(lx,ly)else 0,1 if self.cell_has_trap(rx,ry)else 0,1 if self.cell_has_trap(cx,cy)else 0]
        code=0
        for b in bits:
            code=(code<<1)|b
        return code
    def apply_action(self,action):
        if not self.alive:
            return
        if action==ACTION_LEFT:
            self.agent_dir=(self.agent_dir-1)%4
        elif action==ACTION_RIGHT:
            self.agent_dir=(self.agent_dir+1)%4
        elif action==ACTION_FORWARD:
            dx,dy=DIRS[self.agent_dir]
            self.agent_x,self.agent_y=wrap_pos(self.agent_x+dx,self.agent_y+dy)
        elif action==ACTION_STAY:
            self.stay_count+=1
        if self.cell_has_trap(self.agent_x,self.agent_y):
            self.alive=False
        self.steps+=1
        self.update_traps()

class Automaton:
    def __init__(self,table=None):
        self.state=0
        if table is None:
            self.table=self.random_table()
        else:
            self.table=table
    @staticmethod
    def random_table():
        table=[]
        for _ in range(NUM_STATES*NUM_INPUTS):
            next_state=random.randint(0,NUM_STATES-1)
            action=random.randint(0,3)
            table.append((next_state,action))
        return table
    def reset(self):
        self.state=0
    def step(self,input_code):
        idx=self.state*NUM_INPUTS+input_code
        next_state,action=self.table[idx]
        self.state=next_state
        return action
    def clone(self):
        return Automaton(self.table.copy())

def evaluate_automaton(automaton,episodes=5,max_steps=MAX_STEPS_PER_RUN):
    scores=[]
    for ep in range(episodes):
        env=GameEnvironment(seed=1000+ep+random.randint(0,99999))
        automaton.reset()
        while env.alive and env.steps<max_steps:
            input_code=env.get_input_code()
            action=automaton.step(input_code)
            env.apply_action(action)
        score=env.steps-0.05*env.stay_count
        scores.append(score)
    return sum(scores)/len(scores)

class GeneticAlgorithm:
    def __init__(self,pop_size=30,generations=40,mutation_rate=0.05,elite=2):
        self.pop_size=pop_size
        self.generations=generations
        self.mutation_rate=mutation_rate
        self.elite=elite
        self.history=[]
    def mutate(self,automaton):
        child=automaton.clone()
        new_table=child.table
        for i in range(len(new_table)):
            if random.random()<self.mutation_rate:
                next_state,action=new_table[i]
                if random.random()<0.5:
                    next_state=random.randint(0,NUM_STATES-1)
                else:
                    action=random.randint(0,3)
                new_table[i]=(next_state,action)
        child.table=new_table
        return child
    def crossover(self,a1,a2):
        p=random.randint(1,len(a1.table)-1)
        new_table=a1.table[:p]+a2.table[p:]
        return Automaton(new_table)
    def tournament_select(self,population,fitnesses,k=3):
        idxs=random.sample(range(len(population)),k)
        best=idxs[0]
        for i in idxs[1:]:
            if fitnesses[i]>fitnesses[best]:
                best=i
        return population[best]
    def run(self,log_func=None):
        population=[Automaton() for _ in range(self.pop_size)]
        best_global=None
        best_global_fit=-10**9
        for gen in range(self.generations):
            fitnesses=[evaluate_automaton(ind,episodes=3) for ind in population]
            ranked=sorted(zip(population,fitnesses),key=lambda x:x[1],reverse=True)
            population=[x[0] for x in ranked]
            fitnesses=[x[1] for x in ranked]
            if fitnesses[0]>best_global_fit:
                best_global_fit=fitnesses[0]
                best_global=population[0].clone()
            avg_fit=sum(fitnesses)/len(fitnesses)
            self.history.append((gen,fitnesses[0],avg_fit))
            if log_func:
                log_func(f"[GA] Generation {gen:03d} | best={fitnesses[0]:.2f} | avg={avg_fit:.2f}")
            new_population=[population[i].clone() for i in range(self.elite)]
            while len(new_population)<self.pop_size:
                p1=self.tournament_select(population,fitnesses)
                p2=self.tournament_select(population,fitnesses)
                child=self.crossover(p1,p2)
                child=self.mutate(child)
                new_population.append(child)
            population=new_population
        return best_global,best_global_fit,self.history

class SimulatedAnnealing:
    def __init__(self,iterations=300,temp=50.0,alpha=0.97):
        self.iterations=iterations
        self.temp=temp
        self.alpha=alpha
        self.history=[]
    def neighbor(self,automaton):
        child=automaton.clone()
        idx=random.randint(0,len(child.table)-1)
        next_state,action=child.table[idx]
        if random.random()<0.5:
            next_state=random.randint(0,NUM_STATES-1)
        else:
            action=random.randint(0,3)
        child.table[idx]=(next_state,action)
        return child
    def run(self,log_func=None):
        current=Automaton()
        current_fit=evaluate_automaton(current,episodes=3)
        best=current.clone()
        best_fit=current_fit
        T=self.temp
        for it in range(self.iterations):
            nxt=self.neighbor(current)
            nxt_fit=evaluate_automaton(nxt,episodes=3)
            delta=nxt_fit-current_fit
            if delta>0 or random.random()<math.exp(delta/max(T,1e-6)):
                current=nxt
                current_fit=nxt_fit
            if current_fit>best_fit:
                best=current.clone()
                best_fit=current_fit
            self.history.append((it,best_fit,current_fit))
            if log_func:
                log_func(f"[SA] Iteration {it:03d} | best={best_fit:.2f} | current={current_fit:.2f} | T={T:.4f}")
            T*=self.alpha
        return best,best_fit,self.history

class App:
    def __init__(self,root):
        self.root=root
        self.root.title("Лабораторная работа: конечный автомат и эволюционная оптимизация")
        self.root.geometry("1200x800")
        self.current_automaton=Automaton()
        self.env=GameEnvironment()
        self.current_automaton.reset()
        self.running=False
        self.training_thread=None
        self.setup_ui()
        self.draw_grid()
    def setup_ui(self):
        style=ttk.Style()
        style.theme_use("clam")
        notebook=ttk.Notebook(self.root)
        notebook.pack(fill="both",expand=True)
        self.tab_sim=ttk.Frame(notebook)
        self.tab_train=ttk.Frame(notebook)
        self.tab_logs=ttk.Frame(notebook)
        notebook.add(self.tab_sim,text="Симуляция")
        notebook.add(self.tab_train,text="Обучение")
        notebook.add(self.tab_logs,text="Логи")
        self.setup_sim_tab()
        self.setup_train_tab()
        self.setup_logs_tab()
    def setup_sim_tab(self):
        left=ttk.Frame(self.tab_sim)
        left.pack(side="left",fill="y",padx=10,pady=10)
        right=ttk.Frame(self.tab_sim)
        right.pack(side="right",fill="both",expand=True,padx=10,pady=10)
        ttk.Button(left,text="Шаг",command=self.step_sim).pack(fill="x",pady=5)
        ttk.Button(left,text="Старт / Стоп",command=self.toggle_sim).pack(fill="x",pady=5)
        ttk.Button(left,text="Сброс",command=self.reset_sim).pack(fill="x",pady=5)
        ttk.Button(left,text="Загрузить лучшего",command=self.load_best).pack(fill="x",pady=5)
        info_frame=ttk.LabelFrame(right,text="Информация",padding="10")
        info_frame.pack(fill="both",expand=True)
        self.info_text=scrolledtext.ScrolledText(info_frame,height=10,width=40)
        self.info_text.pack(fill="both",expand=True)
        canvas_frame=ttk.Frame(right)
        canvas_frame.pack(fill="both",expand=True,pady=10)
        self.canvas=tk.Canvas(canvas_frame,width=CANVAS_SIZE,height=CANVAS_SIZE,bg='white')
        self.canvas.pack()
        self.steps_label=ttk.Label(right,text="Шагов: 0")
        self.steps_label.pack()
    def setup_train_tab(self):
        frame=ttk.Frame(self.tab_train,padding="10")
        frame.pack(fill="both",expand=True)
        algo_frame=ttk.LabelFrame(frame,text="Алгоритм",padding="10")
        algo_frame.pack(fill="x",pady=5)
        self.algo_var=tk.StringVar(value="ga")
        ttk.Radiobutton(algo_frame,text="Генетический алгоритм",variable=self.algo_var,value="ga").pack(anchor="w")
        ttk.Radiobutton(algo_frame,text="Имитация отжига",variable=self.algo_var,value="sa").pack(anchor="w")
        params_frame=ttk.LabelFrame(frame,text="Параметры",padding="10")
        params_frame.pack(fill="x",pady=5)
        ttk.Label(params_frame,text="Популяция / Итераций:").grid(row=0,column=0,sticky="w")
        self.pop_size_var=tk.StringVar(value="30")
        ttk.Entry(params_frame,textvariable=self.pop_size_var,width=10).grid(row=0,column=1)
        ttk.Label(params_frame,text="Поколений:").grid(row=1,column=0,sticky="w")
        self.generations_var=tk.StringVar(value="40")
        ttk.Entry(params_frame,textvariable=self.generations_var,width=10).grid(row=1,column=1)
        ttk.Label(params_frame,text="Скорость мутации:").grid(row=2,column=0,sticky="w")
        self.mutation_var=tk.StringVar(value="0.05")
        ttk.Entry(params_frame,textvariable=self.mutation_var,width=10).grid(row=2,column=1)
        btn_frame=ttk.Frame(frame)
        btn_frame.pack(fill="x",pady=10)
        ttk.Button(btn_frame,text="Начать обучение",command=self.start_training).pack(side="left",padx=5)
        ttk.Button(btn_frame,text="Остановить",command=self.stop_training).pack(side="left",padx=5)
        self.progress=ttk.Progressbar(frame,mode='determinate')
        self.progress.pack(fill="x",pady=5)
        self.train_log=scrolledtext.ScrolledText(frame,height=15)
        self.train_log.pack(fill="both",expand=True,pady=5)
    def setup_logs_tab(self):
        frame=ttk.Frame(self.tab_logs,padding="10")
        frame.pack(fill="both",expand=True)
        self.logs_text=scrolledtext.ScrolledText(frame,height=30)
        self.logs_text.pack(fill="both",expand=True)
    def log(self,message,tab="logs"):
        timestamp=time.strftime("%H:%M:%S")
        log_msg=f"[{timestamp}] {message}\n"
        if tab=="logs" and hasattr(self,'logs_text'):
            self.logs_text.insert(tk.END,log_msg)
            self.logs_text.see(tk.END)
        elif tab=="train" and hasattr(self,'train_log'):
            self.train_log.insert(tk.END,log_msg)
            self.train_log.see(tk.END)
        elif tab=="info" and hasattr(self,'info_text'):
            self.info_text.insert(tk.END,log_msg)
            self.info_text.see(tk.END)
    def draw_grid(self):
        self.canvas.delete("all")
        cell_size=CELL_SIZE
        for i in range(GRID_SIZE+1):
            self.canvas.create_line(i*cell_size,0,i*cell_size,CANVAS_SIZE,fill='gray')
            self.canvas.create_line(0,i*cell_size,CANVAS_SIZE,i*cell_size,fill='gray')
    def draw_game(self):
        self.canvas.delete("trap","agent")
        cell_size=CELL_SIZE
        for (x,y),ttl in self.env.traps.items():
            x1,y1=x*cell_size,y*cell_size
            x2,y2=x1+cell_size,y1+cell_size
            intensity=min(255,150+ttl*15)
            color=f'#{intensity:02x}00{intensity//2:02x}'
            self.canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline='darkred',tags="trap")
        x1=self.env.agent_x*cell_size
        y1=self.env.agent_y*cell_size
        x2=x1+cell_size
        y2=y1+cell_size
        self.canvas.create_oval(x1+2,y1+2,x2-2,y2-2,fill='blue',tags="agent")
        cx=self.env.agent_x*cell_size+cell_size//2
        cy=self.env.agent_y*cell_size+cell_size//2
        if self.env.agent_dir==DIR_UP:
            self.canvas.create_line(cx,cy-5,cx,cy+3,fill='white',width=2,tags="agent")
            self.canvas.create_polygon(cx-3,cy-5,cx,cy-8,cx+3,cy-5,fill='white',tags="agent")
        elif self.env.agent_dir==DIR_RIGHT:
            self.canvas.create_line(cx+5,cy,cx-3,cy,fill='white',width=2,tags="agent")
            self.canvas.create_polygon(cx+5,cy-3,cx+8,cy,cx+5,cy+3,fill='white',tags="agent")
        elif self.env.agent_dir==DIR_DOWN:
            self.canvas.create_line(cx,cy+5,cx,cy-3,fill='white',width=2,tags="agent")
            self.canvas.create_polygon(cx-3,cy+5,cx,cy+8,cx+3,cy+5,fill='white',tags="agent")
        elif self.env.agent_dir==DIR_LEFT:
            self.canvas.create_line(cx-5,cy,cx+3,cy,fill='white',width=2,tags="agent")
            self.canvas.create_polygon(cx-5,cy-3,cx-8,cy,cx-5,cy+3,fill='white',tags="agent")
    def step_sim(self):
        input_code=self.env.get_input_code()
        action=self.current_automaton.step(input_code)
        self.env.apply_action(action)
        self.draw_game()
        self.steps_label.config(text=f"Шагов: {self.env.steps}")
        if not self.env.alive:
            self.log(f"Агент погиб на шаге {self.env.steps}","info")
            self.running=False
    def toggle_sim(self):
        if self.running:
            self.running=False
        else:
            self.running=True
            self.run_sim_loop()
    def run_sim_loop(self):
        if self.running and self.env.alive and self.env.steps<MAX_STEPS_PER_RUN:
            self.step_sim()
            self.root.after(50,self.run_sim_loop)
        elif not self.env.alive:
            self.running=False
    def reset_sim(self):
        self.running=False
        self.env.reset()
        self.current_automaton.reset()
        self.draw_game()
        self.steps_label.config(text="Шагов: 0")
        self.log("Симуляция сброшена","info")
    def load_best(self):
        if hasattr(self,'best_automaton') and self.best_automaton:
            self.current_automaton=self.best_automaton.clone()
            self.log("Загружен лучший автомат","info")
            self.reset_sim()
        else:
            self.log("Сначала обучите автомат!","info")
    def training_log_callback(self,message):
        self.log(message,"train")
    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log("Обучение уже запущено!","train")
            return
        algo=self.algo_var.get()
        self.train_log.delete(1.0,tk.END)
        self.log(f"Запуск обучения: {algo.upper()}","train")
        def train():
            try:
                if algo=="ga":
                    pop_size=int(self.pop_size_var.get())
                    generations=int(self.generations_var.get())
                    mutation=float(self.mutation_var.get())
                    ga=GeneticAlgorithm(pop_size=pop_size,generations=generations,mutation_rate=mutation,elite=2)
                    best,fitness,history=ga.run(log_func=self.training_log_callback)
                else:
                    iterations=int(self.pop_size_var.get())*10
                    sa=SimulatedAnnealing(iterations=iterations,temp=50.0,alpha=0.97)
                    best,fitness,history=sa.run(log_func=self.training_log_callback)
                self.best_automaton=best
                self.log(f"\n✅ Обучение завершено! Лучший результат: {fitness:.2f}","train")
            except Exception as e:
                self.log(f"Ошибка: {e}","train")
        self.training_thread=threading.Thread(target=train)
        self.training_thread.daemon=True
        self.training_thread.start()
    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log("Остановка обучения...","train")
            self.training_thread=None

def main():
    root=tk.Tk()
    app=App(root)
    root.mainloop()

if __name__=="__main__":
    main()