import os
import json
import datetime
import calendar
import math
import random
import copy
import requests
import yadisk as Y
from collections import defaultdict
from threading import Timer
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

CID = '2bbef6ae892740418c73d62af8e47366'
CS  = 'b2af829ef1b2452baf85d9d9532c84ac'
RT  = '1:svKccVatTXAPlcLk:1hKw1dEXnFBKQZNpkBD63cAlQM1UwWQxHvDTxfT_k6YT9aGaORbmz7tznGip4rli2vZAqKojNw:y7bb2BfPGWvE-_HiCSIgoA'

(S_IN_D,S_IN_A,S_IN_S,
 S_EX_D,S_EX_A,S_EX_C,
 S_BUD_C,S_BUD_A,
 S_GO_N,S_GO_A,
 S_SAV_A,
 S_REM_T,S_REM_D,
 S_TM_M,S_TM_S,
 S_MON_Y,S_MON_M,
 S_INF_S,S_INF_Y,S_INF_R,S_INF_I,
 S_CAL_Y,
 S_L_D,S_L_A,S_L_T,
 S_H_D,
 S_FP
) = range(27)

UD = 'u_data'
os.makedirs(UD, exist_ok=True)
y = None

def init_y():
    tok = requests.post(
        'https://oauth.yandex.ru/token',
        data={'grant_type':'refresh_token','refresh_token':RT,'client_id':CID,'client_secret':CS}
    ).json().get('access_token')
    if tok:
        yd = Y.YaDisk(token=tok)
        if not yd.exists('/Proj'):
            yd.mkdir('/Proj')
        return yd
    return None

def path(uid):
    return os.path.join(UD, f"{uid}.json")

def load(uid):
    p = path(uid)
    if y and y.exists(f'/Proj/{uid}.json'):
        y.download(f'/Proj/{uid}.json', p)
    if os.path.exists(p):
        return json.load(open(p, 'r', encoding='utf-8'))
    return {'i': [], 'e': [], 'b': {}, 'g': [], 's': 0.0, 'r': [], 'p': []}

def save(uid, d):
    p = path(uid)
    open(p, 'w', encoding='utf-8').write(json.dumps(d, ensure_ascii=False, indent=2))
    if y:
        y.upload(p, f'/Proj/{uid}.json', overwrite=True)

i_sum = lambda d: sum(x['a'] for x in d['i'])
e_sum = lambda d: sum(x['a'] for x in d['e'])

def daily(d):
    m = defaultdict(lambda: {'i': 0, 'e': 0})
    mn = mx = None
    for x in d['i'] + d['e']:
        try:
            dt = datetime.datetime.strptime(x['d'], '%Y-%m-%d').date()
            mn = dt if mn is None or dt < mn else mn
            mx = dt if mx is None or dt > mx else mx
            m[dt]['i' if x in d['i'] else 'e'] += x['a']
        except:
            pass
    if mn is None:
        return [], None
    arr = []
    cum = 0
    for i in range((mx - mn).days + 1):
        day = mn + datetime.timedelta(days=i)
        cum += m[day]['i'] - m[day]['e']
        arr.append((i, cum))
    return arr, mn

def lin(arr, n=30):
    if len(arr) < 2:
        return [], 0, 0, 0
    xs = [i for i, _ in arr]
    ys = [v for _, v in arr]
    mx = sum(xs)/len(xs)
    my = sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = sum((x-mx)**2 for x in xs)
    if den == 0:
        return [], 0, 0, 0
    s = num/den
    b = my - s*mx
    f = [(xs[-1]+i, b+s*(xs[-1]+i)) for i in range(1,n+1)]
    ss = sum((y-my)**2 for y in ys)
    sr = sum((y-(s*x+b))**2 for x,y in zip(xs,ys))
    r2 = 1 - sr/ss if ss else 0
    return f, s, b, r2

def expf(d, n=30):
    arr, _ = daily(d)
    inc = [arr[i][1]-arr[i-1][1] for i in range(1,len(arr))]
    r = sum(inc)/len(inc) if inc else 0
    lv = arr[-1][1] if arr else 0
    return [(i, lv + r*i) for i in range(1,n+1)], r

def polf(d, deg=2, n=30):
    import numpy as np
    arr, _ = daily(d)
    if len(arr) < deg + 1:
        return [], []
    xs = np.array([i for i,_ in arr])
    ys = np.array([v for _,v in arr])
    c = np.polyfit(xs, ys, deg)
    p = np.poly1d(c)
    return [(xs[-1]+i, p(xs[-1]+i)) for i in range(1,n+1)], c

def mcsim(d, days=30, k=5):
    arr, _ = daily(d)
    inc = [arr[i][1]-arr[i-1][1] for i in range(1,len(arr))]
    mu = sum(inc)/len(inc) if inc else 0
    sig = math.sqrt(sum((x-mu)**2 for x in inc)/len(inc)) if inc else 0
    lv = arr[-1][1] if arr else 0
    out = []
    for _ in range(k):
        v = lv
        path = []
        for i in range(days):
            v += mu + random.gauss(0, sig)
            path.append((i+1, v))
        out.append(path)
    return out

def histf(d, h=10, f=10):
    arr, _ = daily(d)
    if len(arr) < h:
        return None
    ch = [arr[-i][1]-arr[-i-1][1] for i in range(1,h)]
    mu = sum(ch)/len(ch)
    cur = arr[-1][1]
    return cur, cur+mu*f, mu, math.sqrt(sum((x-mu)**2 for x in ch)/len(ch))

def nextmo(d):
    m = {}
    for x in d['i'] + d['e']:
        try:
            dt = datetime.datetime.strptime(x['d'], '%Y-%m-%d').date()
            k = (dt.year, dt.month)
            m[k] = m.get(k, 0) + (x['a'] if x in d['i'] else -x['a'])
        except:
            pass
    if not m:
        return None
    lm = max(m)
    return m[lm]*2 + d.get('s',0)

def monthdata(d, mth, yr):
    st = datetime.date(yr, mth, 1)
    ed = datetime.date(yr, mth, calendar.monthrange(yr, mth)[1])
    md = defaultdict(float)
    out = []
    for x in d['i'] + d['e']:
        try:
            dt = datetime.datetime.strptime(x['d'], '%Y-%m-%d').date()
            if st <= dt <= ed:
                md[dt] += x['a'] if x in d['i'] else -x['a']
        except:
            pass
    curr = st
    while curr <= ed:
        out.append((curr.day, md[curr]))
        curr += datetime.timedelta(days=1)
    return out

async def start(update:Update, context:ContextTypes.DEFAULT_TYPE):
    context.user_data['d'] = load(update.effective_user.id)
    await update.message.reply_text(
        "Команды: /add_income, /add_expense, /set_budget, /set_goal, /add_savings, /new_reminder, "
        "/start_timer, /month_analysis, /inflation_calc, /calendar, /large_expense, /history_forecast, "
        "/future_pro, /show_plans, /all_reports, /calc_future, /improve, /stats, /author, /global, /tips, /links"
    )

async def add_income(update, context):
    await update.message.reply_text('Дата YYYY-MM-DD:')
    return S_IN_D

async def in_date(update, context):
    context.user_data['tmp'] = {'d': update.message.text}
    await update.message.reply_text('Сумма:')
    return S_IN_A

async def in_amt(update, context):
    context.user_data['tmp']['a'] = float(update.message.text)
    await update.message.reply_text('Источник:')
    return S_IN_S

async def in_src(update, context):
    d = context.user_data['d']
    d['i'].append(context.user_data['tmp'])
    save(update.effective_user.id, d)
    await update.message.reply_text('Добавлено')
    return ConversationHandler.END

async def add_expense(update, context):
    await update.message.reply_text('Дата YYYY-MM-DD:')
    return S_EX_D

async def ex_date(update, context):
    context.user_data['tmp'] = {'d': update.message.text}
    await update.message.reply_text('Сумма:')
    return S_EX_A

async def ex_amt(update, context):
    context.user_data['tmp']['a'] = float(update.message.text)
    await update.message.reply_text('Категория:')
    return S_EX_C

async def ex_cat(update, context):
    d = context.user_data['d']
    tmp = context.user_data['tmp']
    tmp['c'] = update.message.text
    d['e'].append(tmp)
    save(update.effective_user.id, d)
    await update.message.reply_text('Добавлено')
    return ConversationHandler.END

async def set_budget(update, context):
    await update.message.reply_text('Категория:')
    return S_BUD_C

async def bud_cat(update, context):
    context.user_data['tmp'] = {'c': update.message.text}
    await update.message.reply_text('Сумма:')
    return S_BUD_A

async def bud_amt(update, context):
    d = context.user_data['d']
    key = context.user_data['tmp']['c']
    d['b'][key] = float(update.message.text)
    save(update.effective_user.id, d)
    await update.message.reply_text('Установлено')
    return ConversationHandler.END

async def set_goal(update, context):
    await update.message.reply_text('Название цели:')
    return S_GO_N

async def go_name(update, context):
    context.user_data['tmp'] = {'n': update.message.text}
    await update.message.reply_text('Сумма:')
    return S_GO_A

async def go_amt(update, context):
    d = context.user_data['d']
    d['g'].append({'n': context.user_data['tmp']['n'], 'a': float(update.message.text)})
    save(update.effective_user.id, d)
    await update.message.reply_text('Цель добавлена')
    return ConversationHandler.END

async def add_savings(update, context):
    await update.message.reply_text('Сумма:')
    return S_SAV_A

async def sav_amt(update, context):
    d = context.user_data['d']
    d['s'] += float(update.message.text)
    save(update.effective_user.id, d)
    await update.message.reply_text('Сбережения обновлены')
    return ConversationHandler.END

async def new_reminder(update, context):
    await update.message.reply_text('Текст:')
    return S_REM_T

async def rem_text(update, context):
    context.user_data['tmp'] = {'t': update.message.text}
    await update.message.reply_text('Дата YYYY-MM-DD:')
    return S_REM_D

async def rem_date(update, context):
    d = context.user_data['d']
    d['r'].append({'t': context.user_data['tmp']['t'], 'd': update.message.text})
    save(update.effective_user.id, d)
    await update.message.reply_text('Напоминание добавлено')
    return ConversationHandler.END

async def start_timer(update, context):
    await update.message.reply_text('Минуты:')
    return S_TM_M

async def tim_min(update, context):
    context.user_data['tmp'] = {'m': int(update.message.text)}
    await update.message.reply_text('Секунды:')
    return S_TM_S

async def tim_sec(update, context):
    m = context.user_data['tmp']['m']
    s = int(update.message.text)
    total = m*60 + s
    Timer(total, lambda: context.bot.send_message(chat_id=update.effective_chat.id, text='Время вышло')).start()
    await update.message.reply_text('Таймер запущен')
    return ConversationHandler.END

async def month_analysis(update, context):
    await update.message.reply_text('Год:')
    return S_MON_Y

async def mon_y(update, context):
    context.user_data['tmp'] = {'y': int(update.message.text)}
    await update.message.reply_text('Месяц:')
    return S_MON_M

async def mon_m(update, context):
    yv = context.user_data['tmp']['y']
    mv = int(update.message.text)
    arr = monthdata(context.user_data['d'], mv, yv)
    await update.message.reply_text(f'Анализ {mv}.{yv}: {arr}')
    return ConversationHandler.END

async def inflation_calc(update, context):
    await update.message.reply_text('Сумма:')
    return S_INF_S

async def inf_sum(update, context):
    context.user_data['inf'] = {'s': float(update.message.text)}
    await update.message.reply_text('Лет:')
    return S_INF_Y

async def inf_years(update, context):
    context.user_data['inf']['y'] = int(update.message.text)
    await update.message.reply_text('Ставка %:')
    return S_INF_R

async def inf_rate(update, context):
    context.user_data['inf']['r'] = float(update.message.text)
    await update.message.reply_text('Инфляция %:')
    return S_INF_I

async def inf_out(update, context):
    i = context.user_data['inf']
    fv = i['s']*((1+i['r']/100)**i['y'])
    fi = i['s']*((1+i['i']/100)**i['y'])
    rr = (fv/fi - 1)*100
    await update.message.reply_text(f'Будущая: {fv:.2f}, с инфляцией: {fi:.2f}, доходность: {rr:.2f}%')
    return ConversationHandler.END

async def calendar_view(update, context):
    await update.message.reply_text('Год:')
    return S_CAL_Y

async def cal_year(update, context):
    yv = int(update.message.text)
    msg = ''
    for mn in range(1,13):
        msg += f"{calendar.month_name[mn]} {yv}\n"
        for week in calendar.monthcalendar(yv, mn):
            msg += " ".join(f"{d:2}" for d in week) + "\n"
        msg += "\n"
    await update.message.reply_text(msg)
    return ConversationHandler.END

async def large_expense(update, context):
    await update.message.reply_text('Дата YYYY-MM-DD:')
    return S_L_D

async def lrg_date(update, context):
    context.user_data['tmp'] = {'d': update.message.text}
    await update.message.reply_text('Сумма:')
    return S_L_A

async def lrg_amt(update, context):
    context.user_data['tmp']['a'] = float(update.message.text)
    await update.message.reply_text('Описание:')
    return S_L_T

async def lrg_txt(update, context):
    orig = context.user_data['d']
    d2 = copy.deepcopy(orig)
    le = context.user_data['tmp']
    d2['e'].append({'d': le['d'], 'a': le['a'], 'c': 'large'})
    res1 = histf(orig)
    res2 = histf(d2)
    await update.message.reply_text(f"Без: {res1}\nС:   {res2}")
    return ConversationHandler.END

async def history_forecast(update, context):
    await update.message.reply_text('Дней истории:')
    return S_H_D

async def hist_days(update, context):
    n = int(update.message.text)
    res = histf(context.user_data['d'], h=n, f=n)
    await update.message.reply_text(str(res))
    return ConversationHandler.END

async def future_pro(update, context):
    await update.message.reply_text('Команды: /lin /exp /pol /sim')
    return S_FP

async def fp_lin(update, context):
    arr, _, _, _ = lin(daily(context.user_data['d'])[0])
    await update.message.reply_text(str(arr[:10]))
    return S_FP

async def fp_exp(update, context):
    await update.message.reply_text(str(expf(context.user_data['d'])[0][:10]))
    return S_FP

async def fp_pol(update, context):
    await update.message.reply_text(str(polf(context.user_data['d'])[0][:10]))
    return S_FP

async def fp_sim(update, context):
    await update.message.reply_text(str(mcsim(context.user_data['d'])[0]))
    return S_FP

async def show_plans(update, context):
    await update.message.reply_text("\n".join(str(x) for x in context.user_data['d']['p']) or 'Нет планов')

async def all_reports(update, context):
    d = context.user_data['d']
    await update.message.reply_text(f"Итог: {i_sum(d)-e_sum(d)}, Сб: {d['s']}")

async def calc_future(update, context):
    await update.message.reply_text(str(nextmo(context.user_data['d'])))

async def improve(update, context):
    await update.message.reply_text('Советы по улучшению')

async def stats(update, context):
    d = context.user_data['d']
    ce = defaultdict(float)
    for x in d['e']:
        ce[x.get('c','')] += x['a']
    await update.message.reply_text("\n".join(f"{k}: {v}" for k,v in ce.items()) or 'Нет расходов')

async def author(update, context):
    await update.message.reply_text('Р. Равилов, 2025')

async def global_info(update, context):
    await update.message.reply_text('Глобальные данные')

async def tips(update, context):
    await update.message.reply_text('Советы инвестирования')

async def links(update, context):
    await update.message.reply_text('Ссылки')

def main():
    global y
    y = init_y()
    app = ApplicationBuilder().token('7506072380:AAFYLYMS-QASL5cywkSQBIxHEMyChRdz1mI').build()

    app.add_handler(CommandHandler('start', start))

    convs = [
        ConversationHandler(
            entry_points=[CommandHandler('add_income', add_income)],
            states={
                S_IN_D:[MessageHandler(filters.TEXT, in_date)],
                S_IN_A:[MessageHandler(filters.TEXT, in_amt)],
                S_IN_S:[MessageHandler(filters.TEXT, in_src)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('add_expense', add_expense)],
            states={
                S_EX_D:[MessageHandler(filters.TEXT, ex_date)],
                S_EX_A:[MessageHandler(filters.TEXT, ex_amt)],
                S_EX_C:[MessageHandler(filters.TEXT, ex_cat)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('set_budget', set_budget)],
            states={
                S_BUD_C:[MessageHandler(filters.TEXT, bud_cat)],
                S_BUD_A:[MessageHandler(filters.TEXT, bud_amt)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('set_goal', set_goal)],
            states={
                S_GO_N:[MessageHandler(filters.TEXT, go_name)],
                S_GO_A:[MessageHandler(filters.TEXT, go_amt)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('add_savings', add_savings)],
            states={S_SAV_A:[MessageHandler(filters.TEXT, sav_amt)]},
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('new_reminder', new_reminder)],
            states={
                S_REM_T:[MessageHandler(filters.TEXT, rem_text)],
                S_REM_D:[MessageHandler(filters.TEXT, rem_date)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('start_timer', start_timer)],
            states={
                S_TM_M:[MessageHandler(filters.TEXT, tim_min)],
                S_TM_S:[MessageHandler(filters.TEXT, tim_sec)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('month_analysis', month_analysis)],
            states={
                S_MON_Y:[MessageHandler(filters.TEXT, mon_y)],
                S_MON_M:[MessageHandler(filters.TEXT, mon_m)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('inflation_calc', inflation_calc)],
            states={
                S_INF_S:[MessageHandler(filters.TEXT, inf_sum)],
                S_INF_Y:[MessageHandler(filters.TEXT, inf_years)],
                S_INF_R:[MessageHandler(filters.TEXT, inf_rate)],
                S_INF_I:[MessageHandler(filters.TEXT, inf_out)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('calendar', calendar_view)],
            states={S_CAL_Y:[MessageHandler(filters.TEXT, cal_year)]},
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('large_expense', large_expense)],
            states={
                S_L_D:[MessageHandler(filters.TEXT, lrg_date)],
                S_L_A:[MessageHandler(filters.TEXT, lrg_amt)],
                S_L_T:[MessageHandler(filters.TEXT, lrg_txt)],
            },
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('history_forecast', history_forecast)],
            states={S_H_D:[MessageHandler(filters.TEXT, hist_days)]},
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
        ConversationHandler(
            entry_points=[CommandHandler('future_pro', future_pro)],
            states={S_FP:[
                CommandHandler('lin', fp_lin),
                CommandHandler('exp', fp_exp),
                CommandHandler('pol', fp_pol),
                CommandHandler('sim', fp_sim)
            ]},
            fallbacks=[CommandHandler('cancel', lambda _,__: ConversationHandler.END)]
        ),
    ]

    for conv in convs:
        app.add_handler(conv)

    for cmd, fn in [
        ('show_plans', show_plans), ('all_reports', all_reports),
        ('calc_future', calc_future), ('improve', improve),
        ('stats', stats), ('author', author),
        ('global', global_info), ('tips', tips),
        ('links', links)
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    app.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
