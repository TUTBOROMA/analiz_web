import pygame, yadisk, requests, os, json, math, datetime, calendar, random, time, copy
import numpy as np
from array import array
from pygame.locals import *
from collections import defaultdict

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
icon_path = os.path.join(os.path.dirname(__file__), 'icon.bmp')
icon = pygame.image.load(icon_path)
pygame.display.set_icon(icon)
# Создаем окно с флагом RESIZABLE
sc = pygame.display.set_mode((1200, 900), pygame.RESIZABLE)
pygame.display.set_caption("Финансовый Анализ и Прогноз — Равилов. Р. Р (9-Б, 292)")

dt = {}
u = ''
yad = None
tmr = False
tst = 0
tdur = 0
calc_scroll = 0
ar_scroll = 0
m_y = 0
m_m = 0
inf_data = {}
cal_data = {}
last_active = pygame.time.get_ticks()
le_data = {}
le_result_global = {}
history_days_global = 10

try:
    fnt = pygame.font.Font("OpenSans-Regular.ttf", 28)
    small_fnt = pygame.font.Font("OpenSans-Regular.ttf", 20)
    big_fnt = pygame.font.Font("OpenSans-Regular.ttf", 36)
except:
    fnt = pygame.font.SysFont("Calibri", 28)
    small_fnt = pygame.font.SysFont("Calibri", 20)
    big_fnt = pygame.font.SysFont("Calibri", 36)

def gen_beep(fr=440, d=0.5, vol=0.5):
    sr = 44100
    ns = int(round(d * sr))
    b = array("h")
    for i in range(ns):
        val = int(vol * 32767.0 * math.sin(2 * math.pi * fr * i / sr))
        b.append(val)
    return pygame.mixer.Sound(buffer=b)

snd_beep = gen_beep()

CLIENT_ID = '2bbef6ae892740418c73d62af8e47366'
CLIENT_SECRET = 'b2af829ef1b2452baf85d9d9532c84ac'
REFRESH_TOKEN = '1:svKccVatTXAPlcLk:1hKw1dEXnFBKQZNpkBD63cAlQM1UwWQxHvDTxfT_k6YT9aGaORbmz7tznGip4rli2vZAqKojNw:y7bb2BfPGWvE-_HiCSIgoA'

def get_token(rt):
    url = "https://oauth.yandex.ru/token"
    data = {'grant_type': 'refresh_token', 'refresh_token': rt,
            'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}
    r = requests.post(url, data=data)
    if r.status_code == 200:
        return r.json().get('access_token')
    return None

def check_auth(y):
    try:
        y.get_disk_info()
        return True
    except Exception as e:
        print(e)
        return False

def si(dt):
    return sum(item.get('amount', 0) for item in dt.get('income', []))

def se(dt):
    return sum(item.get('amount', 0) for item in dt.get('expenses', []))

def save_data(y, u, d):
    lp = f"{u}.json"
    dp = f"/Proj/{u}.json"
    try:
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
        if not y.exists("/Proj"):
            y.mkdir("/Proj")
        y.upload(lp, dp, overwrite=True)
        os.remove(lp)
    except Exception as e:
        print("Ошибка сохранения данных:", e)

def read_data(y, u):
    lp = f"{u}.json"
    dp = f"/Proj/{u}.json"
    try:
        if not y.exists(dp):
            return None
        y.download(dp, lp)
        with open(lp, 'r', encoding='utf-8') as f:
            dd = json.load(f)
        os.remove(lp)
        return dd
    except Exception as e:
        print("Ошибка чтения данных:", e)
        return None

def get_login():
    return input("ENTER LOGIN")

class Btn:
    def __init__(self, t, x, y, w, h, a=None):
        self.t = t
        self.r = pygame.Rect(x, y, w, h)
        self.a = a
        self.bg = (70, 130, 180)
        self.hover_bg = (100, 180, 240)
        self.curr_bg = self.bg

    def draw(self, s, font):
        pygame.draw.rect(s, self.curr_bg, self.r, border_radius=10)
        ts = font.render(self.t, True, (255, 255, 255))
        tr = ts.get_rect(center=self.r.center)
        s.blit(ts, tr)

    def hov(self, pos):
        return self.r.collidepoint(pos)

    def clk(self, ev):
        if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
            if self.r.collidepoint(ev.pos) and self.a:
                self.a()

class Inp:
    def __init__(self, x, y, w, h, txt='', pm=''):
        self.r = pygame.Rect(x, y, w, h)
        self.inactive_color = pygame.Color('gray')
        self.active_color = pygame.Color('black')
        self.color = self.inactive_color
        self.txt = txt
        self.font = pygame.font.SysFont("Calibri", 24)
        self.txt_surface = self.font.render(txt, True, self.color)
        self.active = False
        self.placeholder = pm

    def handle(self, ev):
        if ev.type == MOUSEBUTTONDOWN:
            self.active = self.r.collidepoint(ev.pos)
            self.color = self.active_color if self.active else self.inactive_color
        if ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_RETURN:
                tmp = self.txt
                self.txt = ''
                self.txt_surface = self.font.render(self.txt, True, self.color)
                return tmp
            elif ev.key == pygame.K_BACKSPACE:
                self.txt = self.txt[:-1]
            else:
                self.txt += ev.unicode
            self.txt_surface = self.font.render(self.txt, True, self.color)
        return None

    def draw(self, s):
        ph = self.font.render(self.placeholder, True, (0, 0, 0))
        s.blit(ph, (self.r.x, self.r.y - 25))
        s.blit(self.txt_surface, (self.r.x + 5, self.r.y + 5))
        pygame.draw.rect(s, self.color, self.r, 2)
def daily_diff(dt):
    dd = defaultdict(lambda: {'i': 0, 'e': 0})
    mn = None
    mx = None
    for inc in dt.get('income', []):
        try:
            d_inc = datetime.datetime.strptime(inc['date'], "%Y-%m-%d").date()
            dd[d_inc]['i'] += inc['amount']
            mn = d_inc if mn is None or d_inc < mn else mn
            mx = d_inc if mx is None or d_inc > mx else mx
        except:
            continue
    for exp in dt.get('expenses', []):
        try:
            d_exp = datetime.datetime.strptime(exp['date'], "%Y-%m-%d").date()
            dd[d_exp]['e'] += exp['amount']
            mn = d_exp if mn is None or d_exp < mn else mn
            mx = d_exp if mx is None or d_exp > mx else mx
        except:
            continue
    if mn is None or mx is None:
        return [], None
    days = (mx - mn).days + 1
    data = []
    cumulative = 0
    for i in range(days):
        d_day = mn + datetime.timedelta(days=i)
        daily_net = dd[d_day]['i'] - dd[d_day]['e']
        cumulative += daily_net
        data.append((i, cumulative))
    return data, mn

def max_drawdown(daily_data):
    max_val = -float('inf')
    max_dd = 0
    for _, v in daily_data:
        if v > max_val:
            max_val = v
        dd_val = max_val - v
        if dd_val > max_dd:
            max_dd = dd_val
    return max_dd

def adv_lr(daily_data, days_fwd=30):
    if not daily_data or len(daily_data) < 2:
        return [], 0.0, 0.0, 0.0
    xv = [i for i, _ in daily_data]
    yv = [v for _, v in daily_data]
    n = len(xv)
    mx = sum(xv)/n
    my = sum(yv)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xv,yv))
    den = sum((x-mx)**2 for x in xv)
    if abs(den) < 1e-9:
        return [], 0.0, 0.0, 0.0
    slope = num/den
    intercept = my - slope*mx
    ss_tot = sum((y-my)**2 for y in yv)
    ss_res = sum((y - (slope*x+intercept))**2 for x,y in zip(xv,yv))
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0
    lx = xv[-1]
    forecast = [(lx+i, intercept + slope*(lx+i)) for i in range(1, days_fwd+1)]
    return forecast, slope, intercept, r2

def exponential_forecast(dt, n=30):
    daily_data, start_date = daily_diff(dt)
    if not daily_data:
        return None, None
    daily_increments = []
    for i in range(1, len(daily_data)):
        inc = daily_data[i][1] - daily_data[i-1][1]
        daily_increments.append(inc)
    r = sum(daily_increments)/len(daily_increments) if daily_increments else 1.0
    last_val = daily_data[-1][1]
    forecast = [(i, last_val + r*i) for i in range(1, n+1)]
    return forecast, r

def polynomial_forecast(dt, degree=2, days_fwd=30):
    daily_data, start_date = daily_diff(dt)
    if not daily_data or len(daily_data) < degree+1:
        return None, None
    xs = np.array([x for x, _ in daily_data])
    ys = np.array([y for _, y in daily_data])
    coeffs = np.polyfit(xs, ys, degree)
    poly = np.poly1d(coeffs)
    forecast = [(xs[-1]+i, poly(xs[-1]+i)) for i in range(1, days_fwd+1)]
    return forecast, coeffs

def monte_carlo_simulation(dt, days=30, simulations=5):
    daily_data, start_date = daily_diff(dt)
    if not daily_data or len(daily_data) < 2:
        return None
    increments = []
    for i in range(1, len(daily_data)):
        inc = daily_data[i][1] - daily_data[i-1][1]
        increments.append(inc)
    avg_inc = sum(increments)/len(increments) if increments else 0
    vol = math.sqrt(sum((x - avg_inc)**2 for x in increments)/len(increments)) if increments else 0
    last_val = daily_data[-1][1]
    sim_results = []
    for _ in range(simulations):
        val = last_val
        path = []
        for i in range(1, days+1):
            change = avg_inc + random.gauss(0, vol)
            val += change
            path.append((i, val))
        sim_results.append(path)
    return sim_results
def history_based_forecast(dt, history_days=10, forecast_days=10):
    daily_data, start_date = daily_diff(dt)
    if not daily_data or len(daily_data) < history_days:
        return None
    last_history = daily_data[-history_days:]
    changes = [last_history[i][1] - last_history[i-1][1] for i in range(1, len(last_history))]
    if len(changes) == 0:
        return None
    avg_change = sum(changes)/len(changes)
    current_net = daily_data[-1][1]
    predicted_net = current_net + avg_change * forecast_days
    volatility = math.sqrt(sum((c - avg_change)**2 for c in changes)/len(changes)) if changes else 0
    return current_net, predicted_net, avg_change, volatility

def forecast_next_month(dt):
    monthly_balance = {}
    for inc in dt.get('income', []):
        try:
            d = datetime.datetime.strptime(inc['date'], "%Y-%m-%d").date()
            key = (d.year, d.month)
            monthly_balance.setdefault(key, 0)
            monthly_balance[key] += inc['amount']
        except:
            continue
    for exp in dt.get('expenses', []):
        try:
            d = datetime.datetime.strptime(exp['date'], "%Y-%m-%d").date()
            key = (d.year, d.month)
            monthly_balance.setdefault(key, 0)
            monthly_balance[key] -= exp['amount']
        except:
            continue
    if not monthly_balance:
        return None
    last_month = max(monthly_balance.keys())
    last_balance = monthly_balance[last_month]
    savings = dt.get('savings', 0)
    return last_balance * 2 + savings

def get_next_russian_holiday():
    holidays = [
        (1, 1, "Новый Год"),
        (1, 7, "Рождество Христово"),
        (2, 23, "День защитника Отечества"),
        (3, 8, "Международный женский день"),
        (5, 1, "Праздник Весны и Труда"),
        (5, 9, "День Победы"),
        (6, 12, "День России"),
        (9, 1, "День знаний"),
        (10, 5, "День учителя"),
        (11, 4, "День народного единства"),
        (12, 31, "Старый Новый Год")
    ]
    today = datetime.date.today()
    next_hol = None
    min_delta = None
    for m, d, name in holidays:
        try:
            hol = datetime.date(today.year, m, d)
        except ValueError:
            continue
        if hol < today:
            hol = datetime.date(today.year+1, m, d)
        delta = (hol - today).days
        if min_delta is None or delta < min_delta:
            min_delta = delta
            next_hol = (hol, name, delta)
    return next_hol
def dm(dt, month, year):
    data = []
    start_date = datetime.date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime.date(year, month, last_day)
    dd = defaultdict(lambda: 0)
    for inc in dt.get('income', []):
        try:
            d = datetime.datetime.strptime(inc['date'], "%Y-%m-%d").date()
            if start_date <= d <= end_date:
                dd[d] += inc.get('amount', 0)
        except:
            continue
    for exp in dt.get('expenses', []):
        try:
            d = datetime.datetime.strptime(exp['date'], "%Y-%m-%d").date()
            if start_date <= d <= end_date:
                dd[d] -= exp.get('amount', 0)
        except:
            continue
    curr = start_date
    while curr <= end_date:
        data.append((curr.day, dd[curr]))
        curr += datetime.timedelta(days=1)
    return data
def draw_line_chart(surf, data, x, y, w, h, title=''):
    pygame.draw.rect(surf, (0, 0, 0), (x, y, w, h), 2)
    title_surf = pygame.font.SysFont("Calibri", 24).render(title, True, (0, 0, 0))
    surf.blit(title_surf, (x + 5, y + 5))
    if not data:
        no_data = pygame.font.SysFont("Calibri", 24).render("Нет данных", True, (255, 0, 0))
        surf.blit(no_data, (x + w//2 - no_data.get_width()//2, y + h//2))
        return
    vals = [v for _, v in data]
    mx = max(vals)
    mn = min(vals)
    if mx == mn:
        mx += 1
    pts = []
    n = len(data)
    dx = w - 60
    dy = h - 60
    ox = x + 40
    oy = y + 40
    for i, (lx, vv) in enumerate(data):
        rx = ox + (dx * (i / (n - 1)))
        rr = (vv - mn) / (mx - mn)
        ry = oy + dy * (1 - rr)
        pts.append((int(rx), int(ry)))
    # Оси
    pygame.draw.line(surf, (0, 0, 0), (ox, oy+dy), (ox+dx, oy+dy), 2)
    pygame.draw.line(surf, (0, 0, 0), (ox, oy), (ox, oy+dy), 2)
    # Подписи осей
    x_label = small_fnt.render("Дни", True, (0,0,0))
    y_label = small_fnt.render("Баланс", True, (0,0,0))
    surf.blit(x_label, (ox+dx-30, oy+dy+10))
    surf.blit(y_label, (ox-70, oy))
    for i in range(len(pts)-1):
        pygame.draw.line(surf, (50, 50, 200), pts[i], pts[i+1], 3)
    for px, py in pts:
        pygame.draw.circle(surf, (255, 0, 0), (px, py), 5)
def draw_bar_chart(surf, data, x, y, w, h, title=''):
    pygame.draw.rect(surf, (0,0,0), (x, y, w, h), 2)
    title_surf = pygame.font.SysFont("Calibri", 24).render(title, True, (0,0,0))
    surf.blit(title_surf, (x+5, y+5))
    if not data:
        no_data = pygame.font.SysFont("Calibri", 24).render("Нет данных", True, (255,0,0))
        surf.blit(no_data, (x+w//2 - no_data.get_width()//2, y+h//2))
        return
    max_val = max(data.values())
    bar_width = w // (len(data)*2)
    gap = bar_width
    i = 0
    for cat, amt in data.items():
        bar_height = int((amt / max_val) * (h - 40))
        bar_x = x + gap + i*(bar_width+gap)
        bar_y = y + h - bar_height - 20
        pygame.draw.rect(surf, (50,50,200), (bar_x, bar_y, bar_width, bar_height))
        cat_surf = pygame.font.SysFont("Calibri", 16).render(cat, True, (0,0,0))
        surf.blit(cat_surf, (bar_x, y+h-20))
        i += 1
    x_label = small_fnt.render("Категории", True, (0,0,0))
    y_label = small_fnt.render("Суммы", True, (0,0,0))
    surf.blit(x_label, (x + w//2 - 40, y + h + 5))
    surf.blit(y_label, (x - 80, y))
def ai():
    global ca, ibs, idat
    ca = 'add_income'
    ibs = [Inp(400, 100, 300, 40, pm='Дата (YYYY-MM-DD):'),
           Inp(400, 160, 300, 40, pm='Сумма дохода:'),
           Inp(400, 220, 300, 40, pm='Источник:')]
    idat = {}
    ibs[0].active = True

def ae():
    global ca, ibs, idat
    ca = 'add_expense'
    ibs = [Inp(400, 100, 300, 40, pm='Дата (YYYY-MM-DD):'),
           Inp(400, 160, 300, 40, pm='Сумма расхода:'),
           Inp(400, 220, 300, 40, pm='Категория:')]
    idat = {}
    ibs[0].active = True

def sb():
    global ca, ibs, idat
    ca = 'set_budget'
    ibs = [Inp(400, 100, 300, 40, pm='Категория:'),
           Inp(400, 160, 300, 40, pm='Сумма:')]
    idat = {}
    ibs[0].active = True

def sg():
    global ca, ibs, idat
    ca = 'set_goal'
    ibs = [Inp(400, 100, 300, 40, pm='Название цели:'),
           Inp(400, 160, 300, 40, pm='Сумма цели:')]
    idat = {}
    ibs[0].active = True

def asv():
    global ca, ibs, idat
    ca = 'add_savings'
    ibs = [Inp(400, 100, 300, 40, pm='Сумма сбережений:')]
    idat = {}
    ibs[0].active = True

def nrn():
    global ca, ibs, idat
    ca = 'new_reminder'
    ibs = [Inp(400, 100, 300, 40, pm='Текст напоминания:'),
           Inp(400, 160, 300, 40, pm='Дата (YYYY-MM-DD):')]
    idat = {}
    ibs[0].active = True

def stt():
    global ca, ibs, idat
    ca = 'start_timer'
    ibs = [Inp(400, 100, 300, 40, pm='Минуты:'),
           Inp(400, 160, 300, 40, pm='Секунды:')]
    idat = {}
    ibs[0].active = True

def mai():
    global ca, ibs, idat
    ca = 'month_analysis_input'
    ibs = [Inp(400, 100, 300, 40, pm='Год (YYYY):'),
           Inp(400, 160, 300, 40, pm='Месяц (1-12):')]
    idat = {}
    ibs[0].active = True

def ici():
    global ca, ibs, inf_data
    ca = 'inflation_calc_input'
    ibs = [Inp(400, 100, 300, 40, pm='Сумма вклада:'),
           Inp(400, 160, 300, 40, pm='Срок (лет):'),
           Inp(400, 220, 300, 40, pm='Процент вклада:'),
           Inp(400, 280, 300, 40, pm='Инфляция (%):')]
    inf_data = {}
    ibs[0].active = True

def cvi():
    global ca, ibs, cal_data
    ca = 'calendar_view_input'
    ibs = [Inp(400, 100, 300, 40, pm='Год (YYYY):')]
    cal_data = {}
    ibs[0].active = True

# Новая кнопка "Крупная трата"
def kl():
    global ca, ibs, le_data
    ca = 'large_expense_input'
    ibs = [Inp(400, 100, 300, 40, pm='Дата (YYYY-MM-DD):'),
           Inp(400, 160, 300, 40, pm='Сумма крупной траты:'),
           Inp(400, 220, 300, 40, pm='Описание (не обязательно):')]
    le_data = {}
    ibs[0].active = True
def hbi():
    global ca, ibs, idat
    ca = 'history_based_forecast_input'
    ibs = [Inp(400, 100, 300, 40, pm='Количество дней истории:')]
    idat = {}
    ibs[0].active = True

def sp():
    global ca
    ca = 'show_plans'

def ar_():
    global ca
    ca = 'all_reports'

def cf():
    global ca
    ca = 'calc_future'
def hbf():
    hbi()

def fa_pro():
    global ca
    ca = 'future_analysis_pro'

def improve_advice():
    global ca
    ca = 'improve_advice'

def mi():
    global ca
    ca = 'more_info'

def ta():
    global ca
    ca = 'tips_advice'

def ds():
    global ca
    ca = 'data_stats'

def ai_():
    global ca
    ca = 'author_info'

def gi():
    global ca
    ca = 'global_info'

def btm():
    global ca
    ca = 'menu'

def ci():
    global ca, ibs, idat
    ca = 'menu'
    ibs = []
    idat = {}

def fa_linear():
    global ca
    ca = 'future_analysis_linear'

def fa_exponential():
    global ca
    ca = 'future_analysis_exponential'

def fa_polynomial():
    global ca
    ca = 'future_analysis_polynomial'

def fa_simulation():
    global ca
    ca = 'future_analysis_simulation'
def links():
    global ca
    ca = 'links'
def apply_in(i, txt):
    global ca, idat, m_y, m_m, inf_data, cal_data, le_data, history_days_global
    if ca == 'add_income':
        if i == 0:
            idat['date'] = txt
        elif i == 1:
            try:
                idat['amount'] = float(txt)
            except:
                idat['amount'] = 0.0
        elif i == 2:
            idat['source'] = txt
    elif ca == 'add_expense':
        if i == 0:
            idat['date'] = txt
        elif i == 1:
            try:
                idat['amount'] = float(txt)
            except:
                idat['amount'] = 0.0
        elif i == 2:
            idat['category'] = txt
    elif ca == 'set_budget':
        if i == 0:
            idat['cat'] = txt
        elif i == 1:
            try:
                idat['amt'] = float(txt)
            except:
                idat['amt'] = 0.0
    elif ca == 'set_goal':
        if i == 0:
            idat['goal_name'] = txt
        elif i == 1:
            try:
                idat['goal_amt'] = float(txt)
            except:
                idat['goal_amt'] = 0.0
    elif ca == 'add_savings':
        if i == 0:
            try:
                idat['savings'] = float(txt)
            except:
                idat['savings'] = 0.0
    elif ca == 'new_reminder':
        if i == 0:
            idat['rem_text'] = txt
        elif i == 1:
            idat['rem_date'] = txt
    elif ca == 'start_timer':
        if i == 0:
            try:
                idat['timer_min'] = int(txt)
            except:
                idat['timer_min'] = 0
        elif i == 1:
            try:
                idat['timer_sec'] = int(txt)
            except:
                idat['timer_sec'] = 0
    elif ca == 'month_analysis_input':
        if i == 0:
            try:
                idat['year'] = int(txt)
            except:
                idat['year'] = 0
        elif i == 1:
            try:
                idat['month'] = int(txt)
            except:
                idat['month'] = 0
    elif ca == 'inflation_calc_input':
        if i == 0:
            try:
                inf_data['sum'] = float(txt)
            except:
                inf_data['sum'] = 0
        elif i == 1:
            try:
                inf_data['yrs'] = float(txt)
            except:
                inf_data['yrs'] = 1
        elif i == 2:
            try:
                inf_data['dep'] = float(txt)
            except:
                inf_data['dep'] = 0
        elif i == 3:
            try:
                inf_data['inf'] = float(txt)
            except:
                inf_data['inf'] = 0
    elif ca == 'calendar_view_input':
        if i == 0:
            try:
                cal_data['year'] = int(txt)
            except:
                cal_data['year'] = 0
    elif ca == 'large_expense_input':
        if i == 0:
            le_data['date'] = txt
        elif i == 1:
            try:
                le_data['amount'] = float(txt)
            except:
                le_data['amount'] = 0.0
        elif i == 2:
            le_data['description'] = txt
    elif ca == 'history_based_forecast_input':
        try:
            history_days = int(txt)
        except:
            history_days = 10
        history_days_global = history_days
        ca = 'history_based_forecast_result'

def fin():
    global ca, idat, dt, le_data, le_result_global
    if ca == 'add_income':
        r = {'date': idat.get('date', ''),
             'amount': idat.get('amount', 0.0),
             'source': idat.get('source', '')}
        dt.setdefault('income', []).append(r)
    elif ca == 'add_expense':
        r = {'date': idat.get('date', ''),
             'amount': idat.get('amount', 0.0),
             'category': idat.get('category', '')}
        dt.setdefault('expenses', []).append(r)
    elif ca == 'set_budget':
        c = idat.get('cat', '')
        a = idat.get('amt', 0.0)
        dt.setdefault('budgets', {})[c] = a
    elif ca == 'set_goal':
        r = {'name': idat.get('goal_name', ''),
             'amount': idat.get('goal_amt', 0.0)}
        dt.setdefault('goals', []).append(r)
    elif ca == 'add_savings':
        a = idat.get('savings', 0.0)
        dt['savings'] = dt.get('savings', 0.0) + a
    elif ca == 'new_reminder':
        r = {'text': idat.get('rem_text', ''),
             'date': idat.get('rem_date', '')}
        dt.setdefault('reminders', []).append(r)
    elif ca == 'large_expense_input':
        # Симуляция крупной траты
        large_amount = le_data.get('amount', 0.0)
        large_date = le_data.get('date', '')
        description = le_data.get('description', '')
        total_income = si(dt)
        perc = (large_amount / total_income * 100) if total_income != 0 else 0
        dt_without = copy.deepcopy(dt)
        dt_with = copy.deepcopy(dt)
        dt_with.setdefault('expenses', []).append({'date': large_date, 'amount': large_amount, 'category': 'Крупная трата'})
        res_without = history_based_forecast(dt_without, history_days=10, forecast_days=10)
        res_with = history_based_forecast(dt_with, history_days=10, forecast_days=10)
        le_result = {
            'large_amount': large_amount,
            'percentage': perc,
            'forecast_without': res_without,
            'forecast_with': res_with,
            'description': description,
            'date': large_date
        }
        le_result_global = le_result
        ca = 'large_expense_result'

def main():
    global yad, u, dt, ca, ibs, idat, tmr, tst, tdur, calc_scroll, ar_scroll, m_y, m_m, last_active, sc
    tk = get_token(REFRESH_TOKEN)
    if not tk:
        print("Ошибка получения токена.")
        return
    y_ = yadisk.YaDisk(token=tk)
    if not check_auth(y_):
        print("Аутентификация не пройдена.")
        return
    yad = y_
    u = get_login()
    ld = read_data(yad, u)
    if ld:
        dt.update(ld)
    else:
        dt = {'income': [], 'expenses': [], 'budgets': {}, 'goals': [],
              'savings': 0.0, 'reminders': [], 'plans': []}
    current_login = time.time()
    if "last_login" in dt:
        diff_seconds = current_login - dt["last_login"]
    else:
        diff_seconds = None
    dt["last_login"] = current_login
    mbs = [
        Btn("Добавить доход", 50, 120, 300, 50, ai),
        Btn("Добавить расход", 50, 190, 300, 50, ae),
        Btn("Установить бюджет", 50, 260, 300, 50, sb),
        Btn("Установить цель", 50, 330, 300, 50, sg),
        Btn("Добавить сбережения", 50, 400, 300, 50, asv),
        Btn("Планы", 50, 470, 300, 50, sp),
        Btn("Новое напоминание", 50, 540, 300, 50, nrn),
        Btn("Крупная трата", 50, 610, 300, 50, kl),
        Btn("Все отчёты", 50, 680, 300, 50, ar_),
        Btn("Прогноз будущего", 50, 750, 300, 50, cf),
        Btn("Расчет с историей", 50, 820, 300, 50, hbf),
        Btn("Советы по улучшению", 50, 890, 300, 50, improve_advice),
        Btn("Анализ месяца", 400, 120, 300, 50, mai),
        Btn("Кальк. инфляции", 400, 190, 300, 50, ici),
        Btn("Календарь", 400, 260, 300, 50, cvi),
        Btn("Советы и инвестиции", 400, 330, 300, 70, ta),
        Btn("Статистика", 400, 420, 300, 50, ds),
        Btn("Старт таймера", 400, 490, 300, 50, stt),
        Btn("Об авторе", 400, 560, 300, 50, ai_),
        Btn("Глобальная инфо", 400, 630, 300, 50, gi)
    ]
    extra_btns = [Btn("Ссылки", 750, 120, 300, 50, links)]
    extra_btns[0].bg = (0,200,0)
    extra_btns[0].hover_bg = (0,255,0)
    cb = Btn("Отмена", 500, 300, 150, 50, ci)
    bb = Btn("Назад", 1000, 800, 150, 50, btm)
    pro_buttons = [
        Btn("Линейный анализ", 100, 200, 300, 50, fa_linear),
        Btn("Экспоненциальный рост", 100, 270, 300, 50, fa_exponential),
        Btn("Полиномиальный прогноз", 100, 340, 300, 50, fa_polynomial),
        Btn("Симуляция сценариев", 100, 410, 300, 50, fa_simulation)
    ]
    clk = pygame.time.Clock()
    run = True
    timer_beep_played = False
    while run:
        dt_ = clk.tick(30)
        bb.r.x = sc.get_width() - 160
        bb.r.y = sc.get_height() - 70
        for ev in pygame.event.get():
            if ev.type == VIDEORESIZE:
                sc = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            last_active = pygame.time.get_ticks()
            if ev.type == QUIT:
                save_data(yad, u, dt)
                run = False
                break
            if ca in ['show_plans','all_reports','calc_future','more_info',
                      'tips_advice','data_stats','author_info','global_info',
                      'large_expense_result','improve_advice','history_based_forecast_result','links',
                      'future_analysis_pro','future_analysis_linear','future_analysis_exponential',
                      'future_analysis_polynomial','future_analysis_simulation']:
                if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                    if bb.hov(pygame.mouse.get_pos()):
                        bb.clk(ev)
                        ca = 'menu'
            elif ca == 'menu':
                for b in mbs:
                    b.curr_bg = b.hover_bg if b.hov(pygame.mouse.get_pos()) else b.bg
                    b.clk(ev)
                for b in extra_btns:
                    b.curr_bg = b.hover_bg if b.hov(pygame.mouse.get_pos()) else b.bg
                    b.clk(ev)
            elif ca in ['add_income','add_expense','set_budget','set_goal',
                        'add_savings','new_reminder','start_timer','month_analysis_input',
                        'inflation_calc_input','calendar_view_input','large_expense_input','history_based_forecast_input']:
                cb.curr_bg = cb.hover_bg if cb.hov(pygame.mouse.get_pos()) else cb.bg
                cb.clk(ev)
                for ib in ibs:
                    res = ib.handle(ev)
                    if res is not None:
                        i_index = ibs.index(ib)
                        apply_in(i_index, res)
                        ib.active = False
                        if i_index+1 < len(ibs):
                            ibs[i_index+1].active = True
                        else:
                            if ca == 'start_timer':
                                m_ = idat.get('timer_min', 0)
                                s_ = idat.get('timer_sec', 0)
                                tdur = (m_*60+s_)*1000
                                tst = pygame.time.get_ticks()
                                tmr = True
                                timer_beep_played = False
                                ca = 'menu'
                                ibs = []; idat = {}
                            elif ca == 'month_analysis_input':
                                yy = idat.get('year', 0)
                                mm = idat.get('month', 0)
                                if yy>0 and 1<=mm<=12:
                                    m_y = yy; m_m = mm
                                    ca = 'month_analysis_result'
                                else:
                                    ca = 'menu'
                                ibs = []; idat = {}
                            elif ca == 'inflation_calc_input':
                                ca = 'inflation_calc_result'
                                ibs = []
                            elif ca == 'calendar_view_input':
                                ca = 'calendar_view_result'
                                ibs = []
                            elif ca in ['large_expense_input']:
                                fin()
                                ibs = []; idat = {}
                            elif ca in ['history_based_forecast_input']:
                                ibs = []; idat = {}
                            else:
                                fin()
                                ca = 'menu'
                                ibs = []; idat = {}
            elif ca in ['future_analysis_pro']:
                for b in pro_buttons:
                    b.curr_bg = b.hover_bg if b.hov(pygame.mouse.get_pos()) else b.bg
                    b.clk(ev)
                if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                    if bb.hov(pygame.mouse.get_pos()):
                        bb.clk(ev)
            elif ca in ['future_analysis_linear','future_analysis_exponential',
                        'future_analysis_polynomial','future_analysis_simulation']:
                if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                    if bb.hov(pygame.mouse.get_pos()):
                        bb.clk(ev)
                        ca = 'future_analysis_pro'
        sc.fill((245,245,245))
        next_hol = get_next_russian_holiday()
        if next_hol:
            hol_date, hol_name, days_left = next_hol
            hol_text = f"Ближайший праздник: {hol_name} ({hol_date.isoformat()}). До него осталось {days_left} суток."
            hol_surf = small_fnt.render(hol_text, True, (0,100,0))
            hol_x = (sc.get_width() - hol_surf.get_width()) // 2
            hol_y = sc.get_height() - hol_surf.get_height() - 10
            sc.blit(hol_surf, (hol_x, hol_y))
        if ca=='menu':
            title_text = big_fnt.render("Финансовый анализ и прогноз", True, (0,0,0))
            sc.blit(title_text, (sc.get_width()//2 - title_text.get_width()//2, 20))
            if diff_seconds is not None:
                hours = int(diff_seconds // 3600)
                minutes = int((diff_seconds % 3600) // 60)
                seconds = int(diff_seconds % 60)
                last_login_text = f"С момента предыдущего захода: {hours}ч {minutes}м {seconds}с"
            else:
                last_login_text = "Это ваш первый заход"
            sc.blit(small_fnt.render(last_login_text, True, (0, 100, 0)), (50, 80))
            if tmr:
                el = pygame.time.get_ticks()-tst
                if el >= tdur:
                    if not timer_beep_played:
                        snd_beep.play()
                        timer_beep_played = True
                    remaining = 0
                else:
                    remaining = tdur - el
                rs = remaining//1000
                sc.blit(big_fnt.render(f"Таймер: {rs//60}м {rs%60}с", True, (200,0,0)), (900,10))
            for b in mbs:
                b.draw(sc, fnt)
            for b in extra_btns:
                b.draw(sc, fnt)
        elif ca in ['add_income','add_expense','set_budget','set_goal',
                     'add_savings','new_reminder','start_timer','month_analysis_input',
                     'inflation_calc_input','calendar_view_input','large_expense_input','history_based_forecast_input']:
            sc.blit(small_fnt.render("Введите данные и нажмите Enter", True, (0,0,0)), (200,50))
            for ib in ibs:
                ib.draw(sc)
            cb.draw(sc, fnt)
            bb.draw(sc, fnt)
        elif ca=='show_plans':
            y_ = 50
            sc.blit(big_fnt.render("Ваши планы:", True, (0,0,50)), (50,y_))
            y_ += 40
            if not dt.get('plans'):
                sc.blit(small_fnt.render("Планы отсутствуют", True, (0,0,0)), (50,y_))
            else:
                for p in dt['plans']:
                    sc.blit(small_fnt.render(str(p), True, (0,0,0)), (50,y_))
                    y_ += 30
            bb.draw(sc, fnt)
        elif ca=='all_reports':
            sh = 2500
            sf = pygame.Surface((sc.get_width(), sh))
            sf.fill((245,245,245))
            y_ = 20 + ar_scroll
            sf.blit(big_fnt.render("Расширенный анализ:", True, (0,0,50)), (50,y_))
            y_ += 40
            # Основные итоги
            ic = len(dt.get('income', []))
            ec = len(dt.get('expenses', []))
            total_inc = si(dt)
            total_exp = se(dt)
            net = total_inc - total_exp
            sf.blit(small_fnt.render(f"Доходов: {ic}, Расходов: {ec}", True, (0,0,0)), (50,y_))
            y_ += 30
            sf.blit(small_fnt.render(f"Итог: {net:.2f} (Доход: {total_inc:.2f}, Расход: {total_exp:.2f})", True, (0,0,0)), (50,y_))
            y_ += 30
            sf.blit(small_fnt.render(f"Сбережения: {dt.get('savings', 0):.2f}", True, (0,0,0)), (50,y_))
            y_ += 40
            sf.blit(small_fnt.render("Список доходов:", True, (0,0,0)), (50,y_))
            y_ += 25
            for inc in dt.get('income', []):
                sf.blit(small_fnt.render(f"Дата: {inc.get('date','')} | Сумма: {inc.get('amount',0):.2f} | Источник: {inc.get('source','')}", True, (0,0,0)), (70,y_))
                y_ += 20
            y_ += 20
            sf.blit(small_fnt.render("Список расходов:", True, (0,0,0)), (50,y_))
            y_ += 25
            for exp in dt.get('expenses', []):
                sf.blit(small_fnt.render(f"Дата: {exp.get('date','')} | Сумма: {exp.get('amount',0):.2f} | Категория: {exp.get('category','')}", True, (0,0,0)), (70,y_))
                y_ += 20
            y_ += 20
            # Бюджеты
            sf.blit(small_fnt.render("Бюджеты:", True, (0,0,0)), (50,y_))
            y_ += 25
            for cat, amt in dt.get('budgets', {}).items():
                sf.blit(small_fnt.render(f"{cat}: {amt:.2f}", True, (0,0,0)), (70,y_))
                y_ += 20
            y_ += 20
            # Цели
            sf.blit(small_fnt.render("Цели:", True, (0,0,0)), (50,y_))
            y_ += 25
            for goal in dt.get('goals', []):
                sf.blit(small_fnt.render(f"{goal.get('name','')}: {goal.get('amount',0):.2f}", True, (0,0,0)), (70,y_))
                y_ += 20
            y_ += 20
            sf.blit(small_fnt.render("Напоминания:", True, (0,0,0)), (50,y_))
            y_ += 25
            for rem in dt.get('reminders', []):
                sf.blit(small_fnt.render(f"{rem.get('date','')}: {rem.get('text','')}", True, (0,0,0)), (70,y_))
                y_ += 20
            y_ += 20
            # Планы
            sf.blit(small_fnt.render("Планы:", True, (0,0,0)), (50,y_))
            y_ += 25
            for plan in dt.get('plans', []):
                sf.blit(small_fnt.render(str(plan), True, (0,0,0)), (70,y_))
                y_ += 20
            bb.draw(sf, fnt)
            sc.blit(sf, (0,0))
        elif ca=='calc_future':
            sh = 1000
            sf = pygame.Surface((sc.get_width(), sh))
            sf.fill((245,245,245))
            y_ = 20 + calc_scroll
            sf.blit(big_fnt.render("Прогноз будущего:", True, (0,0,50)), (50,y_))
            y_ += 40
            total_inc = si(dt)
            total_exp = se(dt)
            net = total_inc - total_exp
            sf.blit(small_fnt.render(f"Доход: {total_inc:.2f}, Расход: {total_exp:.2f}, Остаток: {net:.2f}", True, (0,0,0)), (50,y_))
            y_ += 30
            sf.blit(small_fnt.render(f"Сбережения: {dt.get('savings', 0):.2f}", True, (0,0,0)), (50,y_))
            y_ += 40
            fc = forecast_next_month(dt)
            if fc is not None:
                sf.blit(small_fnt.render(f"Прогноз на след. месяц: {fc:.2f}", True, (0,0,0)), (50,y_))
            else:
                sf.blit(small_fnt.render("Недостаточно данных для прогноза", True, (255,0,0)), (50,y_))
            bb.draw(sf, fnt)
            sc.blit(sf, (0,0))
        elif ca=='month_analysis_result':
            y_ = 50
            sc.blit(big_fnt.render(f"Анализ {m_m}.{m_y}", True, (0,0,50)), (50,y_))
            y_ += 40
            month_data = dm(dt, m_m, m_y)
            inc_m = sum(v for _, v in month_data if v>0)
            exp_m = sum(-v for _, v in month_data if v<0)
            net_m = inc_m - exp_m
            sc.blit(small_fnt.render(f"Доход: {inc_m:.2f}, Расход: {exp_m:.2f}, Итог: {net_m:.2f}", True, (0,0,0)), (50,y_))
            y_ += 40
            draw_line_chart(sc, month_data, 50, y_, 600, 250, "Дневной накопительный итог")
            y_ += 270
            bb.draw(sc, fnt)
        elif ca=='inflation_calc_result':
            y_ = 50
            sc.blit(big_fnt.render("Калькулятор инфляции:", True, (0,0,50)), (50,y_))
            y_ += 40
            s_val = inf_data.get('sum', 0)
            yrs = inf_data.get('yrs', 1)
            dep = inf_data.get('dep', 0)
            inf = inf_data.get('inf', 0)
            future_dep = s_val * ((1+dep/100)**yrs)
            future_inf = s_val * ((1+inf/100)**yrs)
            real_ret = (future_dep/future_inf - 1) if future_inf != 0 else 0
            sc.blit(small_fnt.render(f"Вклад: {s_val:.2f}, Срок: {yrs} лет, Ставка: {dep:.2f}%, Инфляция: {inf:.2f}%", True, (0,0,0)), (50,y_))
            y_ += 30
            sc.blit(small_fnt.render(f"Будущий вклад: {future_dep:.2f}", True, (0,0,0)), (50,y_))
            y_ += 30
            sc.blit(small_fnt.render(f"С учётом инфляции: {future_inf:.2f}", True, (0,0,0)), (50,y_))
            y_ += 30
            sc.blit(small_fnt.render(f"Реальная доходность: {real_ret*100:.2f}%", True, (0,0,0)), (50,y_))
            bb.draw(sc, fnt)
        elif ca=='calendar_view_result':
            y_ = 50
            sc.blit(big_fnt.render("Календарь на год:", True, (0,0,50)), (50,y_))
            y_ += 40
            year_val = cal_data.get('year', 0)
            if year_val<=0:
                sc.blit(small_fnt.render("Неверный год", True, (255,0,0)), (50,y_))
            else:
                rows, cols = 4, 3
                startX, startY = 50, y_
                cellW, cellH = 350, 250
                month_num = 1
                for r in range(rows):
                    for c in range(cols):
                        if month_num>12:
                            break
                        xx = startX + c*cellW
                        yy_cell = startY + r*cellH
                        m_name = calendar.month_name[month_num]
                        sc.blit(big_fnt.render(f"{m_name} {year_val}", True, (0,0,50)), (xx+10, yy_cell+10))
                        cal_obj = calendar.monthcalendar(year_val, month_num)
                        y_cell = yy_cell+60
                        for week in cal_obj:
                            week_str = "  ".join(f"{day:2}" if day != 0 else "  " for day in week)
                            sc.blit(small_fnt.render(week_str, True, (0,0,0)), (xx+10, y_cell))
                            y_cell += 25
                        month_num += 1
            bb.draw(sc, fnt)
        elif ca=='more_info':
            y_ = 50
            sc.blit(big_fnt.render("Дополнительная информация", True, (0,0,50)), (50,y_))
            y_ += 40
            info_text = [
                "1) Формат дат: YYYY-MM-DD. Используйте этот формат для корректного распознавания.",
                "2) Учитывайте инфляцию при инвестициях – реальные доходы могут отличаться от номинальных.",
                "3) Данные сохраняются на Яндекс.Диске для безопасности.",
                "4) Регулярно анализируйте отчёты для корректировки планов.",
                "5) Волатильность – это мера изменчивости баланса, показывающая риск колебаний доходов и расходов.",
                "6) Инвестиционные советы: диверсифицируйте портфель, откладывайте минимум 15% дохода,",
                "   создавайте резервный фонд на 3-6 месяцев, изучайте рыночные тренды и аналитические отчёты."
            ]
            for line in info_text:
                sc.blit(small_fnt.render(line, True, (0,0,0)), (50,y_))
                y_ += 30
            bb.draw(sc, fnt)
        elif ca=='tips_advice':
            y_ = 50
            sc.blit(big_fnt.render("Инвестиционные советы", True, (0,0,50)), (50,y_))
            y_ += 40
            advice = [
                "1) Откладывайте не менее 15% дохода для формирования капитала.",
                "2) Создавайте резервный фонд, равный 3-6 месяцам расходов.",
                "3) Диверсифицируйте инвестиции для снижения рисков.",
                "4) Регулярно анализируйте рынок, читайте аналитические отчёты и обзоры.",
                "5) Планируйте бюджет и следите за расходами – это поможет избежать долгов.",
                "6) Инвестируйте в образование – знания помогут принимать лучшие финансовые решения.",
                "7) Постоянно совершенствуйте свои стратегии и корректируйте план при необходимости."
            ]
            for line in advice:
                sc.blit(small_fnt.render(line, True, (0,0,0)), (50,y_))
                y_ += 30
            bb.draw(sc, fnt)
        elif ca=='data_stats':
            y_ = 50
            sc.blit(big_fnt.render("Статистика данных", True, (0,0,50)), (50,y_))
            y_ += 40
            cat_exp = defaultdict(float)
            for exp in dt.get('expenses', []):
                cat = exp.get('category', 'Прочее')
                cat_exp[cat] += exp['amount']
            if cat_exp:
                sc.blit(small_fnt.render("Расходы по категориям:", True, (0,0,0)), (50,y_))
                y_ += 30
                for cat, amt in cat_exp.items():
                    sc.blit(small_fnt.render(f"{cat}: {amt:.2f}", True, (0,0,0)), (70,y_))
                    y_ += 25
                draw_bar_chart(sc, cat_exp, 600, 50, 500, 300, "Диаграмма расходов")
            else:
                sc.blit(small_fnt.render("Нет данных по расходам.", True, (0,0,0)), (50,y_))
            bb.draw(sc, fnt)
        elif ca=='author_info':
            y_ = 50
            sc.blit(big_fnt.render("Об авторе", True, (0,0,50)), (50,y_))
            y_ += 40
            author_info = [
                "Автор: Р. Равилов, 9-Б",
                "Версия: Ultimate 1.0",
                "Дата: 2025-02-01",
                "Все права защищены"
            ]
            for line in author_info:
                sc.blit(small_fnt.render(line, True, (0,0,0)), (50,y_))
                y_ += 30
            bb.draw(sc, fnt)
        elif ca=='global_info':
            y_ = 50
            sc.blit(big_fnt.render("Глобальные экономические данные", True, (0,0,50)), (50,y_))
            y_ += 40
            global_info = [
                "Россия: ~6.0% роста ВВП",
                "США: ~2.2% роста",
                "ЕС: ~3.5% роста",
                "Китай: ~2.3% роста",
                "Япония: ~1.8% роста",
                "Совет: анализируйте данные и корректируйте инвестиционные стратегии!"
            ]
            for line in global_info:
                sc.blit(small_fnt.render(line, True, (0,0,0)), (50,y_))
                y_ += 30
            bb.draw(sc, fnt)
        elif ca=='future_analysis_pro':
            sc.blit(big_fnt.render("Продвинутый анализ будущего PRO", True, (0,0,50)), (100,100))
            for b in pro_buttons:
                b.draw(sc, fnt)
            bb.draw(sc, fnt)
        elif ca=='future_analysis_linear':
            daily_data, start_date = daily_diff(dt)
            forecast, slope, intercept, r2 = adv_lr(daily_data)
            sc.blit(big_fnt.render("Линейный анализ будущего", True, (0,0,50)), (100,100))
            sc.blit(small_fnt.render(f"slope = {slope:.2f}  |  R^2 = {r2:.2f}", True, (0,0,0)), (100,160))
            y_ = 200
            for i, (idx, val) in enumerate(forecast[:10], 1):
                pred_date = start_date + datetime.timedelta(days=idx)
                sc.blit(small_fnt.render(f"{pred_date.isoformat()} : {val:.2f}", True, (0,0,0)), (100,y_))
                y_ += 25
            bb.draw(sc, fnt)
        elif ca=='future_analysis_exponential':
            daily_data, start_date = daily_diff(dt)
            forecast, r = exponential_forecast(dt)
            sc.blit(big_fnt.render("Экспоненциальный рост", True, (0,0,50)), (100,100))
            sc.blit(small_fnt.render(f"Средний прирост = {r:.2f}", True, (0,0,0)), (100,160))
            y_ = 200
            for i, (idx, val) in enumerate(forecast[:10], 1):
                pred_date = start_date + datetime.timedelta(days=idx)
                sc.blit(small_fnt.render(f"{pred_date.isoformat()} : {val:.2f}", True, (0,0,0)), (100,y_))
                y_ += 25
            bb.draw(sc, fnt)
        elif ca=='future_analysis_polynomial':
            daily_data, start_date = daily_diff(dt)
            forecast, coeffs = polynomial_forecast(dt, degree=2)
            sc.blit(big_fnt.render("Полиномиальный прогноз", True, (0,0,50)), (100,100))
            sc.blit(small_fnt.render("Коэффициенты: " + ", ".join(f"{c:.2f}" for c in coeffs), True, (0,0,0)), (100,160))
            y_ = 200
            for i, (idx, val) in enumerate(forecast[:10], 1):
                pred_date = start_date + datetime.timedelta(days=idx)
                sc.blit(small_fnt.render(f"{pred_date.isoformat()} : {val:.2f}", True, (0,0,0)), (100,y_))
                y_ += 25
            bb.draw(sc, fnt)
        elif ca=='future_analysis_simulation':
            daily_data, start_date = daily_diff(dt)
            sims = monte_carlo_simulation(dt, days=30, simulations=5)
            sc.blit(big_fnt.render("Симуляция сценариев", True, (0,0,50)), (100,100))
            if sims:
                final_vals = [path[-1][1] for path in sims]
                avg_final = sum(final_vals)/len(final_vals)
                sc.blit(small_fnt.render(f"Окончательное среднее = {avg_final:.2f}", True, (0,0,0)), (100,160))
                y_ = 200
                for idx, path in enumerate(sims, 1):
                    pred_date = start_date + datetime.timedelta(days=path[-1][0])
                    sc.blit(small_fnt.render(f"Сценарий {idx}: {pred_date.isoformat()} = {path[-1][1]:.2f}", True, (0,0,0)), (100,y_))
                    y_ += 25
            else:
                sc.blit(small_fnt.render("Недостаточно данных", True, (255,0,0)), (100,200))
            bb.draw(sc, fnt)
        elif ca == 'history_based_forecast_result':
            daily_data, start_date = daily_diff(dt)
            res = history_based_forecast(dt, history_days=history_days_global, forecast_days=history_days_global)
            if res is None:
                sc.blit(big_fnt.render("Недостаточно данных для расчета истории", True, (255, 0, 0)), (100, 100))
            else:
                current_net, predicted_net, avg_change, volatility = res
                max_dd = max_drawdown(daily_data)
                forecast_ext = []
                if daily_data:
                    last_idx, last_val = daily_data[-1]
                    for i in range(1, history_days_global + 1):
                        forecast_ext.append((last_idx + i, last_val + avg_change * i))
                sc.blit(big_fnt.render("Расчет с историей", True, (0, 0, 50)), (100, 100))
                y_ = 170
                sc.blit(small_fnt.render(f"Прогноз рассчитан на основе {history_days_global} последних дней.", True,
                                         (0, 0, 0)), (100, y_))
                y_ += 30
                sc.blit(small_fnt.render(f"Средний дневной прирост = {avg_change:.2f}", True, (0, 0, 0)), (100, y_))
                y_ += 30
                sc.blit(small_fnt.render(f"Текущий итог = {current_net:.2f}", True, (0, 0, 0)), (100, y_))
                y_ += 30
                sc.blit(small_fnt.render(f"Прогноз через {history_days_global} дней = {predicted_net:.2f}", True,
                                         (0, 0, 0)), (100, y_))
                y_ += 30
                sc.blit(small_fnt.render(f"Волатильность = {volatility:.2f} | Макс. просадка = {max_dd:.2f}", True,
                                         (0, 0, 0)), (100, y_))
                y_ += 30
                sc.blit(small_fnt.render("Волатильность – это мера изменчивости баланса, отражающая риск колебаний доходов и расходов.", True, (0,0,0)), (100, y_))
                y_ += 40
                sc.blit(small_fnt.render("Метод: вычисление среднего прироста за последние дни и экстраполяция.", True,
                                         (0, 0, 0)), (100, y_))
                y_ += 30
                hist_dates = [(start_date + datetime.timedelta(days=i), bal) for i, bal in daily_data]
                forecast_dates = [(start_date + datetime.timedelta(days=i), bal) for i, bal in forecast_ext]
                combined = hist_dates + forecast_dates
                draw_line_chart(sc, [((d - start_date).days, bal) for d, bal in combined], 100, y_, 800, 300,
                                "История + прогноз")
            bb.draw(sc, fnt)
        elif ca == 'large_expense_result':
            res = le_result_global
            sc.blit(big_fnt.render("Анализ крупной траты", True, (0, 0, 50)), (100, 100))
            y_ = 170
            sc.blit(small_fnt.render(
                f"Крупная трата: {res['large_amount']:.2f} (это {res['percentage']:.2f}% от общего дохода)", True,
                (0, 0, 0)), (100, y_))
            y_ += 30
            if res['description']:
                sc.blit(small_fnt.render(f"Описание: {res['description']}", True, (0, 0, 0)), (100, y_))
                y_ += 30
            sc.blit(small_fnt.render("Прогноз без крупной траты:", True, (0, 0, 0)), (100, y_))
            y_ += 30
            if res['forecast_without']:
                current_without, predicted_without, avg_without, vol_without = res['forecast_without']
                sc.blit(
                    small_fnt.render(f"Текущий итог: {current_without:.2f} -> Прогноз: {predicted_without:.2f}", True,
                                     (0, 0, 0)), (100, y_))
                y_ += 30
            else:
                sc.blit(small_fnt.render("Недостаточно данных", True, (255, 0, 0)), (100, y_))
                y_ += 30
            sc.blit(small_fnt.render("Прогноз с крупной тратой:", True, (0, 0, 0)), (100, y_))
            y_ += 30
            if res['forecast_with']:
                current_with, predicted_with, avg_with, vol_with = res['forecast_with']
                sc.blit(small_fnt.render(f"Текущий итог: {current_with:.2f} -> Прогноз: {predicted_with:.2f}", True,
                                         (0, 0, 0)), (100, y_))
                y_ += 30
            else:
                sc.blit(small_fnt.render("Недостаточно данных", True, (255, 0, 0)), (100, y_))
                y_ += 30
            sc.blit(small_fnt.render("Сравнение графиков:", True, (0, 0, 0)), (100, y_))
            y_ += 30
            sc.blit(small_fnt.render("Без крупной траты:", True, (0, 0, 0)), (100, y_))
            y_ += 30
            if res['forecast_without']:
                hist_data = daily_diff(dt)[0]
                draw_line_chart(sc, [(i, bal) for i, bal in hist_data], 100, y_, 350, 250, "Без крупной траты")
            sc.blit(small_fnt.render("С крупной тратой:", True, (0, 0, 0)), (500, y_))
            y_ += 30
            if res['forecast_with']:
                dt_with = copy.deepcopy(dt)
                dt_with.setdefault('expenses', []).append(
                    {'date': res.get('date', ''), 'amount': res['large_amount'], 'category': 'Крупная трата'})
                hist_data_with = daily_diff(dt_with)[0]
                draw_line_chart(sc, [(i, bal) for i, bal in hist_data_with], 500, y_, 350, 250, "С крупной трата")
            bb.draw(sc, fnt)
        elif ca == 'improve_advice':
            if m_y == 0 or m_m == 0:
                sc.blit(big_fnt.render("Нет данных для анализа месяца", True, (255, 0, 0)), (100, 100))
            else:
                month_data = dm(dt, m_m, m_y)
                advice = []
                if month_data:
                    inc = sum(v for _, v in month_data if v > 0)
                    exp = sum(-v for _, v in month_data if v < 0)
                    net = inc - exp
                    vol = max_drawdown(month_data)
                    advice.append("Анализ месяца:")
                    advice.append(f"Накопленный итог: {net:.2f}")
                    advice.append(f"Общий доход: {inc:.2f}, общий расход: {exp:.2f}")
                    advice.append(f"Максимальная просадка: {vol:.2f}")
                    if vol > 0.3 * net:
                        advice.append("Просадка слишком высокая. Рассмотрите сокращение расходов и оптимизацию бюджета.")
                    else:
                        advice.append("Просадка в норме.")
                    if inc < exp:
                        advice.append("Расходы превышают доходы – пересмотрите финансовые приоритеты.")
                    else:
                        advice.append("Бюджет сбалансирован, но всегда можно экономить дополнительно.")
                    advice.append("Рекомендуется увеличить сбережения и диверсифицировать инвестиционный портфель.")
                else:
                    advice = ["Нет данных за выбранный месяц."]
                sc.blit(big_fnt.render("Советы по улучшению", True, (0, 0, 50)), (100, 100))
                y_ = 170
                for line in advice:
                    sc.blit(small_fnt.render(line, True, (0, 0, 0)), (100, y_))
                    y_ += 30
            bb.draw(sc, fnt)
        elif ca=='links':
            y_ = 50
            sc.blit(big_fnt.render("Полезные ссылки и материалы", True, (0,150,0)), (50,y_))
            y_ += 40
            links_text = [
                "1) Investopedia – обширный словарь и статьи по финансам: https://www.investopedia.com",
                "2) BBC Business – новости экономики и бизнеса: https://www.bbc.com/news/business",
                "3) Khan Academy Economics – бесплатные уроки по экономике: https://www.khanacademy.org/economics-finance-domain",
                "4) The Balance – советы по личным финансам: https://www.thebalance.com",
                "5) Financial Times – аналитика мировых рынков: https://www.ft.com",
                "6) Coursera – курсы по финансовой грамотности и экономике: https://www.coursera.org",
                "7) Волатильность – показатель изменчивости финансовых показателей, используемый для оценки риска инвестиций.",
                "8) Регулярно читайте аналитические отчёты и обзоры, чтобы быть в курсе экономических трендов."
            ]
            for line in links_text:
                sc.blit(small_fnt.render(line, True, (0,0,0)), (50,y_))
                y_ += 30
            bb.draw(sc, fnt)
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    ca = 'menu'
    ibs = []
    idat = {}
    main()