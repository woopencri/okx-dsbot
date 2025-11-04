import copy
import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import math
import re
import sqlite3
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
load_dotenv()

# ==================== 常量定义 ====================
HOLD_TOLERANCE = 0.5  # HOLD 信号允许的价差百分比

# ==================== 多模型上下文管理 ====================

AI_PROVIDER = 'deepseek'
AI_MODEL = 'deepseek-chat'
ai_client = None
deepseek_client = None
exchange = None
ACTIVE_CONTEXT: Optional["ModelContext"] = None


class ModelContext:
    """封装单个大模型的运行上下文（AI客户端 + 交易所 + 状态容器）"""

    def __init__(self, key: str, meta: Dict[str, str]):
        self.key = key
        self.display = meta.get('display', key.title())
        self.provider = meta.get('provider', key)
        self.model_name = meta.get('model')
        self.base_url = meta.get('base_url')
        self.ai_client = self._create_ai_client()
        self.exchange = self._create_exchange()
        self.markets = {}
        try:
            markets = self.exchange.load_markets()
            self.markets = {symbol: markets.get(symbol) for symbol in TRADE_CONFIGS if symbol in markets}
        except Exception as e:
            print(f'⚠️ {self.display} 加载市场信息失败: {e}')
        self.signal_history = defaultdict(list)
        self.price_history = defaultdict(list)
        self.position_state = defaultdict(dict)
        self.initial_balance = defaultdict(lambda: None)
        self.initial_total_equity: Optional[float] = None
        self.lock = threading.Lock()
        self.web_data = self._create_web_state()
        self.balance_history: List[Dict[str, float]] = []
        self.start_time = datetime.now()
        self.metrics = {
            'ai_calls': 0,
            'signals_generated': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'ai_errors': 0
        }

    # ---------- 初始化辅助 ----------
    def _create_ai_client(self) -> OpenAI:
        if self.provider == 'qwen':
            api_key = os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise RuntimeError("缺少 DASHSCOPE_API_KEY，用于初始化 Qwen 模型。")
            return OpenAI(api_key=api_key, base_url=self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # 默认 DeepSeek
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，用于初始化 DeepSeek 模型。")
        return OpenAI(api_key=api_key, base_url=self.base_url or "https://api.deepseek.com")

    def _create_exchange(self) -> ccxt.okx:
        suffix = self.key.upper()
        api_key = os.getenv(f'OKX_API_KEY_{suffix}', os.getenv('OKX_API_KEY'))
        secret = os.getenv(f'OKX_SECRET_{suffix}', os.getenv('OKX_SECRET'))
        password = os.getenv(f'OKX_PASSWORD_{suffix}', os.getenv('OKX_PASSWORD'))
        sub_account = os.getenv(f'OKX_SUBACCOUNT_{suffix}')

        if not all([api_key, secret, password]):
            raise RuntimeError(f"缺少 {self.display} 的 OKX API 配置，请设置 OKX_API_KEY_{suffix}/OKX_SECRET_{suffix}/OKX_PASSWORD_{suffix}")

        self.sub_account = sub_account

        client = ccxt.okx({
            'options': {
                'defaultType': 'swap',   # USDⓈ 永续
                'defaultSettle': 'usdt'
            },
            'apiKey': api_key,
            'secret': secret,
            'password': password,
        })

        # 子账户支持
        if sub_account:
            client.headers = client.headers or {}
            client.headers.update({'OK-ACCESS-SUBACCOUNT': sub_account})

        return client

    def _create_web_state(self) -> Dict:
        symbol_states = {
            symbol: {
                'account_info': {},
                'current_position': None,
                'current_price': 0,
                'trade_history': [],
                'ai_decisions': [],
                'performance': {
                    'total_profit': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'current_leverage': config['leverage_default'],
                    'suggested_leverage': config['leverage_default'],
                    'leverage_history': [],
                    'last_order_value': 0,
                    'last_order_quantity': 0,
                    'last_order_contracts': 0,
                    'price_change': 0
                },
            'kline_data': [],
            'profit_curve': [],
            'analysis_records': [],
            'last_update': None
        } for symbol, config in TRADE_CONFIGS.items()
        }

        return {
            'model': self.key,
            'display': self.display,
            'symbols': symbol_states,
            'ai_model_info': {
                'provider': self.provider,
                'model': self.model_name,
                'status': 'unknown',
                'last_check': None,
                'error_message': None
            },
            'account_summary': {
                'total_balance': 0,
                'available_balance': 0,
                'total_equity': 0,
                'total_unrealized_pnl': 0
            },
            'account_info': {},
            'balance_history': []
        }


@contextmanager
def activate_context(ctx: ModelContext):
    """切换全局变量到指定模型上下文，确保旧函数兼容"""
    global exchange, ai_client, deepseek_client, AI_PROVIDER, AI_MODEL, ACTIVE_CONTEXT
    global signal_history, price_history, position_state, web_data, initial_balance

    prev_exchange = exchange
    prev_ai_client = ai_client
    prev_deepseek_client = deepseek_client
    prev_ai_provider = AI_PROVIDER
    prev_ai_model = AI_MODEL
    prev_signal_history = signal_history
    prev_price_history = price_history
    prev_position_state = position_state
    prev_web_data = web_data
    prev_initial_balance = initial_balance
    prev_active_context = ACTIVE_CONTEXT

    try:
        exchange = ctx.exchange
        ai_client = ctx.ai_client
        deepseek_client = ctx.ai_client
        AI_PROVIDER = ctx.provider
        AI_MODEL = ctx.model_name
        signal_history = ctx.signal_history
        price_history = ctx.price_history
        position_state = ctx.position_state
        web_data = ctx.web_data
        initial_balance = ctx.initial_balance
        ACTIVE_CONTEXT = ctx
        yield
    finally:
        exchange = prev_exchange
        ai_client = prev_ai_client
        deepseek_client = prev_deepseek_client
        AI_PROVIDER = prev_ai_provider
        AI_MODEL = prev_ai_model
        signal_history = prev_signal_history
        price_history = prev_price_history
        position_state = prev_position_state
        web_data = prev_web_data
        initial_balance = prev_initial_balance
        ACTIVE_CONTEXT = prev_active_context

# 多交易对配置 - 支持6个交易对同时运行
TRADE_CONFIGS = {
    'BTC/USDT:USDT': {
        'display': 'BTC-USDT',
        'amount': 0.001,  # 最小交易量
        'leverage': 10,  # 默认杠杆
        'leverage_min': 3,
        'leverage_max': 20,
        'leverage_default': 10,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,  # 初始测试模式
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    },
    'ETH/USDT:USDT': {
        'display': 'ETH-USDT',
        'amount': 0.01,
        'leverage': 10,
        'leverage_min': 3,
        'leverage_max': 20,
        'leverage_default': 10,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    },
    'OKB/USDT:USDT': {
        'display': 'OKB-USDT',
        'amount': 1,
        'leverage': 8,
        'leverage_min': 3,
        'leverage_max': 15,
        'leverage_default': 8,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    },
    'SOL/USDT:USDT': {
        'display': 'SOL-USDT',
        'amount': 0.1,
        'leverage': 8,
        'leverage_min': 3,
        'leverage_max': 15,
        'leverage_default': 8,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    },
    'DOGE/USDT:USDT': {
        'display': 'DOGE-USDT',
        'amount': 10,
        'leverage': 5,
        'leverage_min': 3,
        'leverage_max': 10,
        'leverage_default': 5,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    },
    'XRP/USDT:USDT': {
        'display': 'XRP-USDT',
        'amount': 10,
        'leverage': 5,
        'leverage_min': 3,
        'leverage_max': 10,
        'leverage_default': 5,
        'leverage_step': 1,
        'timeframe': '5m',
        'test_mode': False,
        'data_points': 96,
        'analysis_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 96
        }
    }
}

# 币种价格显示精度配置（小数位数）
PRICE_DECIMALS = {
    'DOGE/USDT:USDT': 4,  # DOGE使用4位小数
    'XRP/USDT:USDT': 4,   # XRP使用4位小数
    # 其他币种默认使用2位小数
}

# 单交易对兼容模式（向后兼容）
TRADE_CONFIG = TRADE_CONFIGS['BTC/USDT:USDT']

# 预置占位容器；实际数据由每个模型上下文维护
price_history = defaultdict(list)
signal_history = defaultdict(list)
position_state = defaultdict(dict)
initial_balance = defaultdict(lambda: None)
web_data: Dict = {}

# 概览状态（首页使用），后续在运行时维护
overview_state = {
    'series': [],
    'models': {},
    'aggregate': {}
}

# 线程锁保护共享数据（跨模型共享）
data_lock = threading.Lock()
order_execution_lock = threading.Lock()

# ==================== P0级风险控制增强 ====================

class RiskManager:
    """全局风险管理器（P0-2）"""
    def __init__(self, max_total_risk_pct=0.30, max_symbol_risk_pct=0.15):
        self.max_total_risk_pct = max_total_risk_pct  # 总风险敞口不超过30%
        self.max_symbol_risk_pct = max_symbol_risk_pct  # 单交易对不超过15%
        self.lock = threading.Lock()
    
    def can_open_position(self, exchange_client, symbol: str, trade_value: float, leverage: int) -> bool:
        """检查是否可以开仓（全局风险限制）"""
        with self.lock:
            try:
                # 1. 获取账户权益
                balance = exchange_client.fetch_balance()
                usdt_details = None
                if 'info' in balance and 'data' in balance['info']:
                    for data_item in balance['info']['data']:
                        if 'details' in data_item:
                            for detail in data_item['details']:
                                if detail.get('ccy') == 'USDT':
                                    usdt_details = detail
                                    break
                
                if not usdt_details:
                    print(f"[风险管理器] ⚠️ 无法获取账户详情，放行")
                    return True
                
                total_equity = float(usdt_details.get('eq', 0))
                if total_equity <= 0:
                    print(f"[风险管理器] ❌ 账户权益为0，拒绝开仓")
                    return False
                
                # 2. 计算当前总风险敞口
                total_exposure = 0
                all_positions = exchange_client.fetch_positions()
                for pos in all_positions:
                    if pos['contracts'] and float(pos['contracts']) > 0:
                        pos_value = float(pos['contracts']) * float(pos['contractSize']) * float(pos.get('markPrice', pos.get('entryPrice', 0)))
                        total_exposure += pos_value
                
                # 3. 计算新增风险敞口
                new_total_exposure = total_exposure + trade_value
                total_risk_ratio = new_total_exposure / total_equity
                
                # 4. 全局风险检查
                if total_risk_ratio > self.max_total_risk_pct:
                    print(f"[风险管理器] ❌ 全局风险超限: {total_risk_ratio:.1%} > {self.max_total_risk_pct:.1%}")
                    print(f"[风险管理器]    当前敞口: ${total_exposure:.2f}, 新增: ${trade_value:.2f}, 总权益: ${total_equity:.2f}")
                    return False
                
                # 5. 单交易对风险检查
                symbol_risk_ratio = trade_value / total_equity
                if symbol_risk_ratio > self.max_symbol_risk_pct:
                    print(f"[风险管理器] ❌ {symbol}单品种风险超限: {symbol_risk_ratio:.1%} > {self.max_symbol_risk_pct:.1%}")
                    return False
                
                print(f"[风险管理器] ✅ 风险检查通过 - 总敞口: {total_risk_ratio:.1%}, 单品种: {symbol_risk_ratio:.1%}")
                return True
                
            except Exception as e:
                print(f"[风险管理器] ⚠️ 风险检查异常: {e}，放行")
                import traceback
                traceback.print_exc()
                return True

# 初始化全局风险管理器
risk_manager = RiskManager(max_total_risk_pct=0.30, max_symbol_risk_pct=0.15)

# 数据持久化目录
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
ARCHIVE_DIR = BASE_DIR / 'archives'
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / 'history.db'

# ==================== 模型上下文初始化 ====================

MODEL_METADATA = {
    'deepseek': {
        'display': 'DeepSeek 策略',
        'provider': 'deepseek',
        'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
        'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    },
    'qwen': {
        'display': 'Qwen 策略',
        'provider': 'qwen',
        'model': os.getenv('QWEN_MODEL', 'qwen-max'),
        'base_url': os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    }
}

enabled_models_env = os.getenv('ENABLED_MODELS', 'deepseek,qwen')
ENABLED_MODELS = [m.strip().lower() for m in enabled_models_env.split(',') if m.strip()]

MODEL_CONTEXTS: Dict[str, ModelContext] = {}
for model_key in ENABLED_MODELS:
    if model_key in MODEL_METADATA:
        MODEL_CONTEXTS[model_key] = ModelContext(model_key, MODEL_METADATA[model_key])
    else:
        print(f"⚠️ 未识别的模型标识: {model_key}，已跳过。")

if not MODEL_CONTEXTS:
    raise RuntimeError("未启用任何可用模型，请检查 ENABLED_MODELS 配置。")

MODEL_ORDER = list(MODEL_CONTEXTS.keys())
DEFAULT_MODEL_KEY = MODEL_ORDER[0]
DEFAULT_CONTEXT = MODEL_CONTEXTS[DEFAULT_MODEL_KEY]

# 初始化全局引用，使旧逻辑默认指向第一个模型
ai_client = DEFAULT_CONTEXT.ai_client
deepseek_client = ai_client
exchange = DEFAULT_CONTEXT.exchange
AI_PROVIDER = DEFAULT_CONTEXT.provider
AI_MODEL = DEFAULT_CONTEXT.model_name
price_history = DEFAULT_CONTEXT.price_history
signal_history = DEFAULT_CONTEXT.signal_history
position_state = DEFAULT_CONTEXT.position_state
initial_balance = DEFAULT_CONTEXT.initial_balance
web_data = DEFAULT_CONTEXT.web_data
ACTIVE_CONTEXT = DEFAULT_CONTEXT

# 概览初始状态
overview_state['models'] = {
    key: {
        'display': ctx.display,
        'ai_model_info': ctx.web_data['ai_model_info'],
        'account_summary': ctx.web_data['account_summary'],
        'sub_account': getattr(ctx, 'sub_account', None)
    } for key, ctx in MODEL_CONTEXTS.items()
}

@contextmanager
def activate_context(ctx: ModelContext):
    """切换全局变量到指定模型上下文，确保旧函数兼容"""
    global exchange, ai_client, deepseek_client, AI_PROVIDER, AI_MODEL
    global signal_history, price_history, position_state, web_data, initial_balance

    prev_exchange = exchange
    prev_ai_client = ai_client
    prev_deepseek_client = deepseek_client
    prev_ai_provider = AI_PROVIDER
    prev_ai_model = AI_MODEL
    prev_signal_history = signal_history
    prev_price_history = price_history
    prev_position_state = position_state
    prev_web_data = web_data
    prev_initial_balance = initial_balance

    try:
        exchange = ctx.exchange
        ai_client = ctx.ai_client
        deepseek_client = ctx.ai_client
        AI_PROVIDER = ctx.provider
        AI_MODEL = ctx.model_name
        signal_history = ctx.signal_history
        price_history = ctx.price_history
        position_state = ctx.position_state
        web_data = ctx.web_data
        initial_balance = ctx.initial_balance
        yield
    finally:
        exchange = prev_exchange
        ai_client = prev_ai_client
        deepseek_client = prev_deepseek_client
        AI_PROVIDER = prev_ai_provider
        AI_MODEL = prev_ai_model
        signal_history = prev_signal_history
        price_history = prev_price_history
        position_state = prev_position_state
        web_data = prev_web_data
        initial_balance = prev_initial_balance


# ==================== 辅助函数 ====================

def get_symbol_config(symbol: str) -> dict:
    """返回指定交易对的配置字典"""
    return TRADE_CONFIGS.get(symbol, TRADE_CONFIG)


def ensure_symbol_state(symbol: str) -> None:
    """初始化缺失的 web_data / position_state / history 容器"""
    with data_lock:
        if symbol not in web_data['symbols']:
            config = get_symbol_config(symbol)
            web_data['symbols'][symbol] = {
                'account_info': {},
                'current_position': None,
                'current_price': 0,
                'trade_history': [],
                'ai_decisions': [],
                'performance': {
                    'total_profit': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'current_leverage': config['leverage_default'],
                    'suggested_leverage': config['leverage_default'],
                    'leverage_history': [],
                    'last_order_value': 0,
                    'last_order_quantity': 0,
                    'last_order_contracts': 0,
                    'price_change': 0
                },
                'kline_data': [],
                'profit_curve': [],
                'last_update': None
            }


def clamp_value(value, min_val, max_val):
    """限制值在范围内"""
    return max(min_val, min(value, max_val))


def round_to_step(value, step):
    """四舍五入到指定步长"""
    return round(value / step) * step


def check_stop_loss_take_profit(symbol: str, current_price: float, config: dict) -> bool:
    """检查止损止盈（P0-1）
    
    返回True表示触发了止损或止盈，已平仓
    返回False表示未触发
    """
    try:
        ctx = get_active_context()
        current_position = get_current_position(symbol)
        
        if not current_position:
            return False
        
        # 查找最近一次开仓信号（包含止损止盈价格）
        history = ctx.signal_history.get(symbol, [])
        if not history:
            return False
        
        last_signal = None
        for record in reversed(history):
            if record.get('signal') in ['BUY', 'SELL']:
                last_signal = record
                break
        
        if not last_signal:
            return False
        
        stop_loss = last_signal.get('stop_loss', 0)
        take_profit = last_signal.get('take_profit', 0)
        
        if not stop_loss or not take_profit:
            return False
        
        side = current_position.get('side')
        entry_price = current_position.get('entry_price', 0)
        
        # 多头持仓检查
        if side == 'long':
            if current_price <= stop_loss:
                print(f"[{config['display']}] 🛑 触发止损！当前价格 ${current_price:.4f} <= 止损价 ${stop_loss:.4f}")
                print(f"[{config['display']}]    入场价: ${entry_price:.4f}, 亏损: {((current_price - entry_price) / entry_price * 100):.2f}%")
                close_position(symbol, "stop_loss", config)
                return True
            elif current_price >= take_profit:
                print(f"[{config['display']}] 🎯 触发止盈！当前价格 ${current_price:.4f} >= 止盈价 ${take_profit:.4f}")
                print(f"[{config['display']}]    入场价: ${entry_price:.4f}, 盈利: {((current_price - entry_price) / entry_price * 100):.2f}%")
                close_position(symbol, "take_profit", config)
                return True
        
        # 空头持仓检查
        elif side == 'short':
            if current_price >= stop_loss:
                print(f"[{config['display']}] 🛑 触发止损！当前价格 ${current_price:.4f} >= 止损价 ${stop_loss:.4f}")
                print(f"[{config['display']}]    入场价: ${entry_price:.4f}, 亏损: {((entry_price - current_price) / entry_price * 100):.2f}%")
                close_position(symbol, "stop_loss", config)
                return True
            elif current_price <= take_profit:
                print(f"[{config['display']}] 🎯 触发止盈！当前价格 ${current_price:.4f} <= 止盈价 ${take_profit:.4f}")
                print(f"[{config['display']}]    入场价: ${entry_price:.4f}, 盈利: {((entry_price - current_price) / entry_price * 100):.2f}%")
                close_position(symbol, "take_profit", config)
                return True
        
        return False
        
    except Exception as e:
        print(f"[{config.get('display', symbol)}] ⚠️ 止损止盈检查异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def close_position(symbol: str, reason: str, config: dict):
    """平仓（P0-1辅助函数）"""
    try:
        position = get_current_position(symbol)
        if not position:
            print(f"[{config['display']}] ⚠️ 无持仓，无需平仓")
            return
        
        contracts = float(position.get('size', 0))
        if contracts <= 0:
            print(f"[{config['display']}] ⚠️ 持仓数量为0，无需平仓")
            return
        
        side = position.get('side')
        
        # 确定平仓方向（多头平仓用sell，空头平仓用buy）
        close_side = 'sell' if side == 'long' else 'buy'
        
        print(f"[{config['display']}] 📤 执行平仓 - 原因: {reason}")
        print(f"[{config['display']}]    方向: {side} -> {close_side}")
        print(f"[{config['display']}]    数量: {contracts:.6f} 张")
        
        # 使用全局锁保护订单执行
        with order_execution_lock:
            exchange.create_market_order(
                symbol, 
                close_side, 
                contracts,
                params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE'}
            )
        
        print(f"[{config['display']}] ✅ 平仓完成: {reason}")
        
        # 更新统计
        ctx = get_active_context()
        ctx.metrics['trades_closed'] += 1
        
        # 记录平仓历史
        time.sleep(1)
        updated_position = get_current_position(symbol)
        
        with data_lock:
            trade_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'CLOSE',
                'side': close_side,
                'price': 0,  # 市价平仓，实际价格由交易所确定
                'amount': 0,
                'contracts': contracts,
                'leverage': position.get('leverage', 1),
                'confidence': 'AUTO',
                'reason': f'自动平仓: {reason}'
            }
            web_data['symbols'][symbol]['trade_history'].append(trade_record)
            if len(web_data['symbols'][symbol]['trade_history']) > 100:
                web_data['symbols'][symbol]['trade_history'].pop(0)
            
            web_data['symbols'][symbol]['current_position'] = updated_position
        
    except Exception as e:
        print(f"[{config['display']}] ❌ 平仓失败: {e}")
        import traceback
        traceback.print_exc()


def detect_abnormal_market(symbol: str, price_data: dict, config: dict) -> tuple:
    """检测异常市场（P0-3）
    
    返回: (is_abnormal: bool, reason: str)
    """
    try:
        technical_data = price_data.get('technical_data', {})
        
        # 1. 价格剧烈波动检查（5分钟变化超过5%）
        price_change = abs(price_data.get('price_change', 0))
        if price_change > 5:
            return True, f"价格5分钟变化超过5%: {price_change:+.2f}%"
        
        # 2. 成交量异常检查（成交量超过均值5倍）
        volume_ratio = technical_data.get('volume_ratio', 1)
        if volume_ratio > 5:
            return True, f"成交量异常放大{volume_ratio:.1f}倍"
        
        # 3. RSI极端值检查
        rsi = technical_data.get('rsi', 50)
        if rsi > 90:
            return True, f"RSI极度超买: {rsi:.1f}"
        elif rsi < 10:
            return True, f"RSI极度超卖: {rsi:.1f}"
        
        # 4. 布林带极端突破检查
        bb_position = technical_data.get('bb_position', 0.5)
        if bb_position > 1.5:
            return True, f"价格极端突破布林带上轨: {bb_position:.2f}"
        elif bb_position < -0.5:
            return True, f"价格极端跌破布林带下轨: {bb_position:.2f}"
        
        # 5. 价格跳空检查（与前一周期价格差异超过3%）
        if 'df' in price_data and len(price_data['df']) >= 2:
            df = price_data['df']
            last_close = df['close'].iloc[-2] if len(df) >= 2 else 0
            current_close = df['close'].iloc[-1]
            if last_close > 0:
                gap_pct = abs((current_close - last_close) / last_close * 100)
                if gap_pct > 3:
                    return True, f"价格跳空{gap_pct:.2f}%"
        
        return False, None
        
    except Exception as e:
        print(f"[{config.get('display', symbol)}] ⚠️ 异常市场检测失败: {e}")
        return False, None


def get_symbol_market(symbol: str) -> Dict:
    ctx = get_active_context()
    market = ctx.markets.get(symbol)
    if not market:
        try:
            ctx.exchange.load_markets()
            market = ctx.exchange.market(symbol)
            ctx.markets[symbol] = market
        except Exception as e:
            print(f"⚠️ {ctx.display} 无法获取 {symbol} 市场信息: {e}")
            market = {}
    return market or {}


def get_symbol_contract_specs(symbol: str) -> Dict[str, float]:
    """返回合约相关规格（contractSize、最小张数等）"""
    market = get_symbol_market(symbol)
    contract_size = market.get('contractSize') or market.get('contract_size') or 1
    try:
        contract_size = float(contract_size)
    except (TypeError, ValueError):
        contract_size = 1.0

    limits = (market.get('limits') or {}).get('amount') or {}
    market_min_contracts = limits.get('min')
    try:
        market_min_contracts = float(market_min_contracts) if market_min_contracts is not None else None
    except (TypeError, ValueError):
        market_min_contracts = None

    config = get_symbol_config(symbol)
    config_min_base = float(config.get('amount', 0) or 0)
    config_min_contracts = (config_min_base / contract_size) if contract_size else config_min_base

    candidates = [value for value in (market_min_contracts, config_min_contracts) if value and value > 0]
    min_contracts = max(candidates) if candidates else 0.0
    min_base = min_contracts * contract_size if contract_size else config_min_base

    precision = (market.get('precision') or {}).get('amount') if market else None
    step = None
    if precision is not None:
        try:
            step = 10 ** (-precision)
        except Exception:
            step = None
    elif market:
        step = market.get('amountIncrement') or market.get('lot')
        try:
            step = float(step) if step else None
        except (TypeError, ValueError):
            step = None

    return {
        'contract_size': contract_size if contract_size else 1.0,
        'min_contracts': min_contracts,
        'min_base': min_base if min_base else config_min_base,
        'precision': precision,
        'step': step
    }


def get_symbol_min_contracts(symbol: str) -> float:
    """最小下单张数"""
    specs = get_symbol_contract_specs(symbol)
    return specs['min_contracts']


def get_symbol_min_amount(symbol: str) -> float:
    specs = get_symbol_contract_specs(symbol)
    config_min = get_symbol_config(symbol).get('amount', 0)
    min_base = specs['min_base'] if specs['min_base'] else config_min
    return max(min_base, config_min)


def get_symbol_amount_precision(symbol: str):
    specs = get_symbol_contract_specs(symbol)
    return specs['precision'], specs['step']


def base_to_contracts(symbol: str, base_quantity: float) -> float:
    """基础量 -> 合约张数"""
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs['contract_size'] if specs else 1.0
    if not contract_size:
        contract_size = 1.0
    return base_quantity / contract_size


def contracts_to_base(symbol: str, contracts: float) -> float:
    """合约张数 -> 基础数量"""
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs['contract_size'] if specs else 1.0
    if not contract_size:
        contract_size = 1.0
    return contracts * contract_size


def adjust_quantity_to_precision(symbol: str, quantity: float, round_up: bool = False) -> float:
    """在基础数量层面调整到合约精度"""
    contracts = base_to_contracts(symbol, quantity)
    contracts = adjust_contract_quantity(symbol, contracts, round_up=round_up)
    return contracts_to_base(symbol, contracts)


def adjust_contract_quantity(symbol: str, contracts: float, round_up: bool = False) -> float:
    ctx = get_active_context()
    precision, step = get_symbol_amount_precision(symbol)
    adjusted = contracts
    if round_up and step:
        adjusted = math.ceil(adjusted / step) * step
    elif round_up:
        adjusted = math.ceil(adjusted)
    try:
        adjusted = float(ctx.exchange.amount_to_precision(symbol, adjusted))
    except Exception:
        if precision is not None:
            factor = 10 ** precision
            if round_up:
                adjusted = math.ceil(adjusted * factor) / factor
            else:
                adjusted = math.floor(adjusted * factor) / factor
    return adjusted


def format_number(value, decimals: int = 2) -> str:
    if value is None:
        return "--"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    formatted = f"{val:.{decimals}f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted


def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "--"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def format_currency(value: Optional[float], decimals: int = 2) -> str:
    """格式化货币数值，值为空时返回 --"""
    if value is None:
        return "--"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${val:,.{decimals}f}"


def get_price_decimals(symbol: str) -> int:
    """获取币种的价格显示精度（小数位数）"""
    return PRICE_DECIMALS.get(symbol, 2)  # 默认2位小数


def format_sequence(values: List[float], indent: int = 2, per_line: int = 10, decimals: int = 2) -> str:
    if not values:
        return " " * indent + "[]"
    parts = [format_number(v, decimals) for v in values]
    lines = []
    for i in range(0, len(parts), per_line):
        chunk = ", ".join(parts[i:i + per_line])
        lines.append(chunk)
    if not lines:
        return " " * indent + "[]"
    result_lines = []
    result_lines.append(" " * indent + "[" + lines[0] + ("," if len(lines) > 1 else "]"))
    for idx in range(1, len(lines)):
        suffix = "," if idx < len(lines) - 1 else "]"
        result_lines.append(" " * (indent + 1) + lines[idx] + suffix)
    return "\n".join(result_lines)


def evaluate_signal_result(signal: str, price_change_pct: float) -> bool:
    signal = (signal or "").upper()
    if signal == "BUY":
        return price_change_pct >= 0
    if signal == "SELL":
        return price_change_pct <= 0
    if signal == "HOLD":
        return abs(price_change_pct) <= HOLD_TOLERANCE
    return False


def update_signal_validation(symbol: str, current_price: float, timestamp: str) -> None:
    ctx = get_active_context()
    history = ctx.signal_history[symbol]
    updated = False
    for record in history:
        if record.get('validation_price') is None and record.get('entry_price'):
            entry_price = record['entry_price']
            if entry_price:
                change_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                change_pct = 0.0
            record['validation_price'] = current_price
            record['validation_timestamp'] = timestamp
            record['price_change_pct'] = change_pct
            result = evaluate_signal_result(record.get('signal'), change_pct)
            record['result'] = 'success' if result else 'fail'
            updated = True
    if updated:
        ctx.web_data['symbols'][symbol]['analysis_records'] = history[-100:]


def compute_accuracy_metrics(history: List[Dict]) -> Dict:
    evaluated = [rec for rec in history if rec.get('result') in ('success', 'fail')]

    def summarize(records: List[Dict]) -> Dict:
        total = len(records)
        success = sum(1 for r in records if r.get('result') == 'success')
        ratio = success / total if total else None
        return {'total': total, 'success': success, 'ratio': ratio}

    metrics = {
        'windows': {
            '10': summarize(evaluated[-10:]),
            '30': summarize(evaluated[-30:]),
            '50': summarize(evaluated[-50:])
        },
        'by_signal': {},
        'by_confidence': {},
        'by_leverage': {}
    }

    for signal_label in ['BUY', 'SELL', 'HOLD']:
        metrics['by_signal'][signal_label] = summarize([r for r in evaluated if r.get('signal') == signal_label])

    for confidence in ['HIGH', 'MEDIUM', 'LOW']:
        metrics['by_confidence'][confidence] = summarize([r for r in evaluated if r.get('confidence') == confidence])

    leverage_buckets = {
        '3-8x': lambda lev: 3 <= lev <= 8,
        '9-12x': lambda lev: 9 <= lev <= 12,
        '13-20x': lambda lev: 13 <= lev <= 20
    }
    for label, predicate in leverage_buckets.items():
        metrics['by_leverage'][label] = summarize([
            r for r in evaluated
            if isinstance(r.get('leverage'), (int, float)) and predicate(int(r['leverage']))
        ])
    return metrics


def format_ratio(summary: Dict) -> str:
    total = summary.get('total', 0)
    success = summary.get('success', 0)
    ratio = summary.get('ratio')
    if not total:
        return "-- (--/0)"
    percent = f"{ratio * 100:.0f}%"
    return f"{percent} ({success}✓/{total})"


def format_history_table(history: List[Dict]) -> str:
    if not history:
        return "  无历史信号记录\n"
    last_records = history[-50:]
    total = len(last_records)
    lines = ["  序号 信号  信心 杠杆  入场价  验证价  涨跌    结果"]
    for idx, record in enumerate(last_records):
        seq_no = idx - total
        signal = (record.get('signal') or '--').upper().ljust(4)
        confidence = (record.get('confidence') or '--').upper().ljust(3)
        leverage_value = record.get('leverage')
        leverage = f"{int(leverage_value) if leverage_value is not None else 0:>2}x"
        entry = format_number(record.get('entry_price'))
        validation = format_number(record.get('validation_price'))
        change_pct = format_percentage(record.get('price_change_pct'))
        result_symbol = {'success': '✓', 'fail': '✗'}.get(record.get('result'), '·')
        lines.append(f"  {seq_no:>3}  {signal} {confidence} {leverage:>4}  {entry:>7}  {validation:>7}  {change_pct:>6}   {result_symbol}")
    return "\n".join(lines)


def format_accuracy_summary(metrics: Dict) -> str:
    lines = ["  【准确率统计分析】", "", "  时间窗口:"]
    lines.append(f"  - 最近10次: {format_ratio(metrics['windows']['10'])}")
    lines.append(f"  - 最近30次: {format_ratio(metrics['windows']['30'])}")
    lines.append(f"  - 最近50次: {format_ratio(metrics['windows']['50'])}")
    lines.append("")
    lines.append("  按信号类型:")
    for signal_label in ['BUY', 'SELL', 'HOLD']:
        lines.append(f"  - {signal_label:<4}: {format_ratio(metrics['by_signal'][signal_label])}")
    lines.append("")
    lines.append("  按信心等级:")
    for confidence in ['HIGH', 'MEDIUM', 'LOW']:
        lines.append(f"  - {confidence:<6}: {format_ratio(metrics['by_confidence'][confidence])}")
    lines.append("")
    lines.append("  按杠杆范围:")
    for bucket in ['3-8x', '9-12x', '13-20x']:
        lines.append(f"  - {bucket:<6}: {format_ratio(metrics['by_leverage'][bucket])}")
    return "\n".join(lines)


def build_position_suggestion_table(position_suggestions: Dict[str, Dict], config: Dict, asset_name: str, symbol: str = None) -> str:
    lines = []
    leverage_min = config['leverage_min']
    leverage_default = config['leverage_default']
    leverage_max = config['leverage_max']
    min_quantity = position_suggestions.get('min_quantity', config['amount'])
    min_contracts = position_suggestions.get('min_contracts', 0)
    
    # 获取价格显示精度
    price_decimals = get_price_decimals(symbol) if symbol else 2

    def row(confidence_label: str, leverage: int) -> str:
        key = f"{confidence_label}_{leverage}"
        suggestion = position_suggestions.get(key, {})
        quantity = suggestion.get('quantity', 0)
        contracts = suggestion.get('contracts')
        value = suggestion.get('value', 0)
        margin = suggestion.get('margin', 0)
        meets_min = suggestion.get('meets_min', True)
        meets_margin = suggestion.get('meets_margin', True)
        status_parts = []
        status_parts.append('满足最小交易量' if meets_min else '低于最小交易量')
        status_parts.append('保证金充足' if meets_margin else '保证金不足')
        flag = '✅' if suggestion.get('meets', True) else '❌'
        status = ' & '.join(status_parts)
        contracts_info = f"{contracts:.3f}张, " if contracts is not None else ""
        return f"  • {leverage}x: {quantity:.6f} {asset_name} ({contracts_info}价值 ${value:,.2f}), 需 {margin:.2f} USDT {flag} {status}"

    lines.append("  【智能仓位建议表】- 已为你精确计算")
    lines.append("")
    usable_margin = position_suggestions.get('usable_margin', position_suggestions.get('available_balance', 0) * 0.8)
    lines.append(
        f"  账户状态: 可用 {position_suggestions.get('available_balance', 0):.2f} USDT | 可用保证金 {usable_margin:.2f} USDT | 价格 ${position_suggestions.get('current_price', 0):,.{price_decimals}f} | 最小量 {min_quantity} {asset_name} ({min_contracts:.3f} 张)"
    )
    lines.append("")
    sections = [
        ('HIGH', '高信心(HIGH) - 70%保证金'),
        ('MEDIUM', '中信心(MEDIUM) - 50%保证金'),
        ('LOW', '低信心(LOW) - 30%保证金')
    ]
    for confidence_key, title in sections:
        lines.append(f"  {title}:")
        for lev in [leverage_min, leverage_default, leverage_max]:
            lines.append(row(confidence_key, lev))
        lines.append("")
    return "\n".join(lines)


def build_professional_prompt(ctx: ModelContext,
                              symbol: str,
                              price_data: Dict,
                              config: Dict,
                              position_suggestions: Dict[str, Dict],
                              sentiment_text: str,
                              current_position: Optional[Dict]) -> str:
    df: pd.DataFrame = price_data.get('full_data')  # type: ignore
    short_df = df.tail(20) if df is not None else None

    prices = short_df['close'].tolist() if short_df is not None else []
    sma5 = short_df['sma_5'].tolist() if short_df is not None else []
    sma20 = short_df['sma_20'].tolist() if short_df is not None else []
    rsi = short_df['rsi'].tolist() if short_df is not None else []
    macd = short_df['macd'].tolist() if short_df is not None else []
    volume = short_df['volume'].tolist() if short_df is not None else []

    history = ctx.signal_history[symbol]
    metrics = compute_accuracy_metrics(history)
    history_table = format_history_table(history)
    accuracy_summary = format_accuracy_summary(metrics)

    runtime_minutes = int((datetime.now() - ctx.start_time).total_seconds() / 60)
    runtime_hours = runtime_minutes / 60
    ai_calls = ctx.metrics['ai_calls']
    open_positions = sum(1 for pos in ctx.position_state.values() if pos)
    closed_trades = ctx.metrics['trades_closed']

    asset_name = config['display'].split('-')[0]
    
    # 获取币种价格显示精度
    price_decimals = get_price_decimals(symbol)
    
    position_table = build_position_suggestion_table(position_suggestions, config, asset_name, symbol)

    if current_position:
        position_status = f"{current_position.get('side', '--')} {current_position.get('size', 0)} {asset_name} @{format_number(current_position.get('entry_price'), price_decimals)}, 未实现盈亏: {format_number(current_position.get('unrealized_pnl'))} USDT"
    else:
        position_status = "无持仓"

    prompt_sections = [
        f"\n  你是专业的加密货币交易分析师 | {config['display']} {config['timeframe']}周期\n",
        f"\n  【系统运行状态】\n  运行时长: {runtime_minutes}分钟 ({runtime_hours:.1f}小时) | AI分析: {ai_calls}次 | 开仓: {ctx.metrics['trades_opened']}次 | 平仓: {closed_trades}次 | 当前持仓: {open_positions}个\n",
        "  ⚠️ 重要: 以下所有时间序列数据按 最旧→最新 排列\n",
        "  【短期序列】最近20周期 = 100分钟 (最旧→最新)\n",
        "  价格 (USDT):\n" + format_sequence(prices, decimals=price_decimals),
        "\n  5周期均线:\n" + format_sequence(sma5, decimals=price_decimals),
        "\n  20周期均线:\n" + format_sequence(sma20, decimals=price_decimals),
        "\n  RSI (14周期):\n" + format_sequence(rsi, decimals=2),
        "\n  MACD:\n" + format_sequence(macd, decimals=2),
        "\n  成交量 (" + asset_name + "):\n" + format_sequence(volume, decimals=2),
        "\n  【你的历史判断验证】最近50次 (最旧→最新)\n" + history_table + "\n",
        accuracy_summary + "\n",
        "  【当前市场状况】\n",
        f"  当前价格: ${price_data['price']:,.{price_decimals}f}\n"
        f"  当前持仓: {position_status}\n"
        f"  市场情绪: {sentiment_text or '暂无数据'}\n",
        "  技术状态:\n"
        f"  - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')}\n"
        f"  - 中期趋势: {price_data['trend_analysis'].get('medium_term', 'N/A')}\n"
        f"  - RSI: {price_data['technical_data'].get('rsi', 0):.2f}\n"
        f"  - MACD: {price_data['technical_data'].get('macd', 0):.2f}\n",
        position_table,
        "  【决策要求】\n"
        "  1️⃣ 综合分析20周期技术指标 + 50次历史验证 + 统计规律\n"
        "  2️⃣ 特别关注: 高信心信号准确率，合理选择杠杆\n"
        "  3️⃣ 当前持仓需要评估是否加仓/减仓/平仓\n"
        "  4️⃣ 从建议表选择匹配的【数量】，禁止自行计算\n",
        "  请用JSON格式返回:\n"
        "  {\n"
        "    \"signal\": \"BUY|SELL|HOLD\",\n"
        "    \"reason\": \"结合20周期趋势+历史准确率的分析(50字内)\",\n"
        "    \"stop_loss\": 具体价格,\n"
        "    \"take_profit\": 具体价格,\n"
        "    \"confidence\": \"HIGH|MEDIUM|LOW\",\n"
        "    \"leverage\": 3-20范围整数,\n"
        "    \"order_quantity\": 从建议表复制的6位小数\n"
        "  }\n"
        "  ---"
    ]

    return "\n".join(prompt_sections)


def append_signal_record(symbol: str, signal_data: Dict, entry_price: float, timestamp: Optional[str] = None) -> Dict:
    ctx = get_active_context()
    history = ctx.signal_history[symbol]
    record = {
        'timestamp': timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'signal': (signal_data.get('signal') or '').upper(),
        'confidence': (signal_data.get('confidence') or 'MEDIUM').upper(),
        'leverage': int(signal_data.get('leverage', 0)) if signal_data.get('leverage') is not None else None,
        'entry_price': entry_price,
        'validation_price': None,
        'validation_timestamp': None,
        'price_change_pct': None,
        'result': None,
        'reason': signal_data.get('reason'),
        'stop_loss': signal_data.get('stop_loss'),
        'take_profit': signal_data.get('take_profit')
    }
    history.append(record)
    if len(history) > 200:
        history.pop(0)
    ctx.web_data['symbols'][symbol]['analysis_records'] = list(history[-100:])
    return record


def setup_exchange():
    """设置交易所参数 - 多交易对版本"""
    try:
        # 为所有交易对设置杠杆
        for symbol, config in TRADE_CONFIGS.items():
            try:
                exchange.set_leverage(
                    config['leverage_default'],
                    symbol,
                    {'mgnMode': 'cross'}  # 全仓模式
                )
                print(f"✓ {config['display']}: 杠杆 {config['leverage_default']}x")
            except Exception as e:
                print(f"✗ {config['display']}: 杠杆设置失败 - {e}")

        # 获取余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        total_equity = balance['USDT']['total']

        # 更新账户摘要
        with data_lock:
            web_data['account_summary'].update({
                'total_balance': usdt_balance,
                'available_balance': usdt_balance,
                'total_equity': total_equity
            })

        print(f"\n💰 当前USDT余额: {usdt_balance:.2f}")
        print(f"💰 总权益: {total_equity:.2f}\n")

        return True
    except Exception as e:
        print(f"❌ 交易所设置失败: {e}")
        return False


def capture_balance_snapshot(ctx: ModelContext, timestamp: Optional[str] = None) -> Optional[Dict[str, float]]:
    """抓取并缓存当前账户余额信息"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        balance = exchange.fetch_balance()
        usdt_info = balance.get('USDT') or {}
        available = float(usdt_info.get('free') or usdt_info.get('available', 0) or 0)
        total_equity = float(usdt_info.get('total') or usdt_info.get('equity', 0) or 0)
        unrealized = float(usdt_info.get('unrealizedPnl', 0) or 0)
    except Exception as e:
        print(f"[{ctx.display}] ⚠️ 获取余额失败: {e}")
        return None

    snapshot = {
        'timestamp': timestamp,
        'available_balance': available,
        'total_equity': total_equity,
        'unrealized_pnl': unrealized,
        'currency': 'USDT'
    }

    with data_lock:
        ctx.web_data['account_summary'].update({
            'total_balance': available,
            'available_balance': available,
            'total_equity': total_equity,
            'total_unrealized_pnl': unrealized
        })

        ctx.web_data.setdefault('balance_history', []).append(snapshot)
        if len(ctx.web_data['balance_history']) > 1000:
            ctx.web_data['balance_history'].pop(0)

        ctx.balance_history.append(snapshot)
        if len(ctx.balance_history) > 5000:
            ctx.balance_history.pop(0)

    history_store.append_balance(ctx.key, snapshot)

    return snapshot


def refresh_overview_from_context(ctx: ModelContext):
    """同步单个模型的账户摘要与AI状态到概览数据"""
    overview_state['models'][ctx.key] = {
        'display': ctx.display,
        'ai_model_info': ctx.web_data['ai_model_info'],
        'account_summary': ctx.web_data['account_summary'],
        'sub_account': getattr(ctx, 'sub_account', None)
    }


def record_overview_point(timestamp: Optional[str] = None):
    """记录所有模型的总金额，用于首页曲线"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    point = {'timestamp': timestamp}
    total_equity = 0.0

    for key, ctx in MODEL_CONTEXTS.items():
        equity = ctx.web_data['account_summary'].get('total_equity', 0) or 0
        point[key] = float(equity)
        total_equity += equity

    overview_state['series'].append(point)
    if len(overview_state['series']) > 500:
        overview_state['series'].pop(0)

    ratios = {}
    if total_equity > 0:
        for key in MODEL_CONTEXTS.keys():
            ratios[key] = point[key] / total_equity

    overview_state['aggregate'] = {
        'timestamp': timestamp,
        'total_equity': total_equity,
        'ratios': ratios
    }


# ==================== 历史数据存储 ====================


class HistoryStore:
    """负责持久化余额历史并提供导出/压缩能力"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self.last_archive_date = self._load_last_archive_date()

    # ---- 基础设施 ----
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS balance_history (
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_equity REAL,
                    available_balance REAL,
                    unrealized_pnl REAL,
                    currency TEXT,
                    PRIMARY KEY (model, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def _load_last_archive_date(self):
        with self._get_conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'last_archive_date'").fetchone()
            if row and row['value']:
                return datetime.strptime(row['value'], '%Y-%m-%d').date()
        return None

    def _update_last_archive_date(self, day):
        with self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('last_archive_date', ?)", (day.strftime('%Y-%m-%d'),))

    # ---- 写入与读取 ----
    def append_balance(self, model: str, snapshot: Dict[str, float]):
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO balance_history(model, timestamp, total_equity, available_balance, unrealized_pnl, currency)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model,
                    snapshot['timestamp'],
                    snapshot.get('total_equity'),
                    snapshot.get('available_balance'),
                    snapshot.get('unrealized_pnl'),
                    snapshot.get('currency', 'USDT')
                )
            )

    def load_recent_balance(self, model: str, limit: int = 500) -> List[Dict[str, float]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (model, limit)
            ).fetchall()
        data = [
            {
                'timestamp': row['timestamp'],
                'total_equity': row['total_equity'],
                'available_balance': row['available_balance'],
                'unrealized_pnl': row['unrealized_pnl'],
                'currency': row['currency']
            }
            for row in reversed(rows)
        ]
        return data

    def fetch_balance_range(self, model: str, start_ts: str, end_ts: str) -> List[Dict[str, float]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
                (model, start_ts, end_ts)
            ).fetchall()
        return [
            {
                'timestamp': row['timestamp'],
                'total_equity': row['total_equity'],
                'available_balance': row['available_balance'],
                'unrealized_pnl': row['unrealized_pnl'],
                'currency': row['currency']
            }
            for row in rows
        ]

    # ---- 存档与导出 ----
    def compress_day(self, day):
        """将指定日期的数据导出为 Excel"""
        day_str = day.strftime('%Y-%m-%d')
        start = f"{day_str} 00:00:00"
        end = f"{day_str} 23:59:59"

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT model, timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY model, timestamp
                """,
                (start, end)
            ).fetchall()

        if not rows:
            return False

        df = pd.DataFrame([dict(row) for row in rows])
        output_path = ARCHIVE_DIR / f"balances-{day.strftime('%Y%m%d')}.xlsx"
        df.to_excel(output_path, index=False)
        self._update_last_archive_date(day)
        self.last_archive_date = day
        return True

    def compress_if_needed(self, current_dt: datetime):
        """每日零点后压缩前一日数据"""
        target_day = current_dt.date() - timedelta(days=1)
        if target_day <= datetime(1970, 1, 1).date():
            return
        if self.last_archive_date and target_day <= self.last_archive_date:
            return
        self.compress_day(target_day)

    def export_range_to_excel(self, start_date: str, end_date: str, output_path: Path, models: Optional[List[str]] = None):
        models = models or MODEL_ORDER
        with self._get_conn() as conn:
            placeholder = ",".join("?" for _ in models)
            query = f"""
                SELECT model, timestamp, total_equity, available_balance, unrealized_pnl, currency
                FROM balance_history
                WHERE model IN ({placeholder}) AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            rows = conn.execute(query, (*models, start_date, end_date)).fetchall()

        if not rows:
            raise ValueError("选定时间范围内没有历史数据可导出。")

        df = pd.DataFrame([dict(row) for row in rows])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)

    def get_latest_before(self, model: str, timestamp: str):
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT timestamp, total_equity, available_balance, unrealized_pnl
                FROM balance_history
                WHERE model = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (model, timestamp)
            ).fetchone()
        return dict(row) if row else None


# 历史数据存储
history_store = HistoryStore(DB_PATH)

for key in MODEL_ORDER:
    ctx = MODEL_CONTEXTS[key]
    loaded_history = history_store.load_recent_balance(ctx.key, limit=1000)
    if loaded_history:
        ctx.balance_history = loaded_history
        ctx.web_data['balance_history'] = list(loaded_history)
        last_point = loaded_history[-1]
        ctx.web_data['account_summary'].update({
            'total_balance': last_point.get('available_balance', 0),
            'available_balance': last_point.get('available_balance', 0),
            'total_equity': last_point.get('total_equity', 0),
            'total_unrealized_pnl': last_point.get('unrealized_pnl', 0)
        })


# ==================== 对外访问辅助 ====================


def list_model_keys() -> List[str]:
    return MODEL_ORDER


def get_model_metadata() -> List[Dict[str, str]]:
    return [
        {
            'key': key,
            'display': ctx.display,
            'model_name': ctx.model_name,
            'provider': ctx.provider,
            'sub_account': getattr(ctx, 'sub_account', None)
        }
        for key, ctx in MODEL_CONTEXTS.items()
    ]


def get_models_status() -> List[Dict[str, Dict]]:
    statuses = []
    for key in MODEL_ORDER:
        ctx = MODEL_CONTEXTS[key]
        with ctx.lock:
            statuses.append({
                'key': key,
                'display': ctx.display,
                'model_name': ctx.model_name,
                'provider': ctx.provider,
                'sub_account': getattr(ctx, 'sub_account', None),
                'ai_model_info': copy.deepcopy(ctx.web_data['ai_model_info']),
                'account_summary': copy.deepcopy(ctx.web_data['account_summary'])
            })
    return statuses


def get_model_snapshot(model_key: str) -> Dict:
    ctx = MODEL_CONTEXTS.get(model_key)
    if not ctx:
        raise KeyError(f"未知模型: {model_key}")

    with ctx.lock:
        snapshot = copy.deepcopy(ctx.web_data)
        snapshot['model'] = ctx.key
        snapshot['display'] = ctx.display
        snapshot['signal_history'] = {
            symbol: list(records)
            for symbol, records in ctx.signal_history.items()
        }
    return snapshot


RANGE_PRESETS = {
    '1d': timedelta(days=1),
    '7d': timedelta(days=7),
    '15d': timedelta(days=15),
    '1m': timedelta(days=30),
    '1y': timedelta(days=365)
}


def resolve_time_range(range_key: str, now: Optional[datetime] = None):
    now = now or datetime.now()
    if range_key == 'all':
        start = datetime(1970, 1, 1)
    else:
        delta = RANGE_PRESETS.get(range_key, RANGE_PRESETS['7d'])
        start = now - delta
    return start.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d %H:%M:%S')


def get_overview_payload(range_key: str = '1d') -> Dict:
    start_ts, end_ts = resolve_time_range(range_key)
    series_by_model: Dict[str, List[Dict[str, float]]] = {}
    aggregate_series_map: Dict[str, Dict[str, float]] = {}

    for key in MODEL_ORDER:
        data = history_store.fetch_balance_range(key, start_ts, end_ts)
        if not data:
            # 如果该范围内无数据，则使用内存中的最后一条
            data = MODEL_CONTEXTS[key].balance_history[-200:]
        formatted = [
            {
                'timestamp': item['timestamp'],
                'total_equity': item['total_equity'],
                'available_balance': item['available_balance'],
                'unrealized_pnl': item.get('unrealized_pnl')
            }
            for item in data
        ]
        series_by_model[key] = formatted

        for point in formatted:
            ts = point['timestamp']
            bucket = aggregate_series_map.setdefault(ts, {})
            bucket[key] = point['total_equity']

    aggregate_series = []
    for ts in sorted(aggregate_series_map.keys()):
        entry = {'timestamp': ts}
        for key in MODEL_ORDER:
            entry[key] = aggregate_series_map[ts].get(key)
        aggregate_series.append(entry)

    models_summary = {}
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for key in MODEL_ORDER:
        ctx = MODEL_CONTEXTS[key]
        latest = history_store.get_latest_before(key, now_str) or {
            'total_equity': ctx.web_data['account_summary'].get('total_equity', 0),
            'available_balance': ctx.web_data['account_summary'].get('available_balance', 0),
            'unrealized_pnl': ctx.web_data['account_summary'].get('total_unrealized_pnl', 0),
            'timestamp': now_str
        }

        base = history_store.get_latest_before(key, start_ts)
        change_abs = None
        change_pct = None
        if base and base.get('total_equity'):
            change_abs = latest['total_equity'] - base['total_equity']
            change_pct = change_abs / base['total_equity'] if base['total_equity'] else None

        models_summary[key] = {
            'display': ctx.display,
            'model_name': ctx.model_name,
            'provider': ctx.provider,
            'sub_account': getattr(ctx, 'sub_account', None),
            'latest_equity': latest['total_equity'],
            'available_balance': latest.get('available_balance', 0),
            'unrealized_pnl': latest.get('unrealized_pnl', 0),
            'change_abs': change_abs,
            'change_pct': change_pct
        }

    total_equity = sum(models_summary[key]['latest_equity'] for key in MODEL_ORDER)
    model_ratios = {}
    if total_equity:
        for key in MODEL_ORDER:
            model_ratios[key] = models_summary[key]['latest_equity'] / total_equity

    return {
        'range': range_key,
        'series': series_by_model,
        'aggregate_series': aggregate_series,
        'models': models_summary,
        'aggregate': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_equity': total_equity,
            'ratios': model_ratios
        }
    }


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}


def get_sentiment_indicators(token="BTC"):
    """获取情绪指标 - 支持多币种版本

    Args:
        token: 币种代码，如 "BTC", "ETH", "SOL" 等
    """
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "b54bcf4d-1bca-4e8e-9a24-22ff2c3d76d5"

        # 获取最近4小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # 只保留核心指标
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": [token]  # 🆕 支持动态指定币种
        }

        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = requests.post(API_URL, json=request_body, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                # 查找第一个有有效数据的时间段
                for period in time_periods:
                    period_data = period.get("data", [])

                    sentiment = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = item.get("value", "").strip()

                        if value:  # 只处理非空值
                            try:
                                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                    sentiment[endpoint] = float(value)
                                    valid_data_found = True
                            except (ValueError, TypeError):
                                continue

                    # 如果找到有效数据
                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment['CO-A-02-01']
                        negative = sentiment['CO-A-02-02']
                        net_sentiment = positive - negative

                        # 正确的时间延迟计算
                        data_delay = int((datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                        print(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")

                        return {
                            'positive_ratio': positive,
                            'negative_ratio': negative,
                            'net_sentiment': net_sentiment,
                            'data_time': period['startTime'],
                            'data_delay_minutes': data_delay
                        }

                print("❌ 所有时间段数据都为空")
                return None

        return None
    except Exception as e:
        print(f"情绪指标获取失败: {e}")
        return None


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}


def get_symbol_ohlcv_enhanced(symbol, config):
    """增强版：获取交易对K线数据并计算技术指标（多交易对版本）"""
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(symbol, config['timeframe'],
                                     limit=config['data_points'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        return {
            'symbol': symbol,
            'display': config['display'],
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': config['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"[{config.get('display', symbol)}] 获取K线数据失败: {e}")
        return None


# 向后兼容函数
def get_btc_ohlcv_enhanced():
    """向后兼容：获取BTC K线数据"""
    return get_symbol_ohlcv_enhanced('BTC/USDT:USDT', TRADE_CONFIGS['BTC/USDT:USDT'])


def generate_technical_analysis_text(price_data, symbol=None):
    """生成技术分析文本"""
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value and pd.notna(value) else default
    
    # 获取价格显示精度
    price_decimals = get_price_decimals(symbol) if symbol else 2

    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - 5周期: {safe_float(tech['sma_5']):.{price_decimals}f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
    - 20周期: {safe_float(tech['sma_20']):.{price_decimals}f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
    - 50周期: {safe_float(tech['sma_50']):.{price_decimals}f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.{price_decimals}f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.{price_decimals}f}
    """
    return analysis_text


def get_current_position(symbol=None):
    """获取当前持仓情况 - OKX版本（多交易对）"""
    try:
        # 默认使用BTC，向后兼容
        if symbol is None:
            symbol = 'BTC/USDT:USDT'

        positions = exchange.fetch_positions([symbol])

        for pos in positions:
            if pos['symbol'] == symbol:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    config = get_symbol_config(symbol)
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else config.get('leverage_default', 10),
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"[{symbol}] 获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 尝试提取JSON代码块（如果AI包在```json```中）
            if '```json' in json_str:
                start = json_str.find('```json') + 7
                end = json_str.find('```', start)
                if end != -1:
                    json_str = json_str[start:end].strip()
            elif '```' in json_str:
                start = json_str.find('```') + 3
                end = json_str.find('```', start)
                if end != -1:
                    json_str = json_str[start:end].strip()
            
            # 尝试直接解析
            try:
                return json.loads(json_str)
            except:
                pass
            
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原始内容: {json_str[:200]}")
            print(f"错误详情: {e}")
            return None


def test_ai_connection(model_key: Optional[str] = None):
    """测试一个或多个AI模型的连接状态"""
    targets = []
    if model_key:
        ctx = MODEL_CONTEXTS.get(model_key)
        if ctx:
            targets.append(ctx)
        else:
            raise KeyError(f"未找到模型: {model_key}")
    else:
        targets = [MODEL_CONTEXTS[key] for key in MODEL_ORDER]

    results = {}

    for ctx in targets:
        with activate_context(ctx):
            try:
                print(f"🔍 测试 {ctx.display} ({ctx.model_name}) 连接...")
                response = ai_client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    timeout=10.0
                )

                if response and response.choices:
                    ctx.web_data['ai_model_info']['status'] = 'connected'
                    ctx.web_data['ai_model_info']['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ctx.web_data['ai_model_info']['error_message'] = None
                    print(f"✓ {ctx.display} 连接正常")
                    results[ctx.key] = True
                else:
                    ctx.web_data['ai_model_info']['status'] = 'error'
                    ctx.web_data['ai_model_info']['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ctx.web_data['ai_model_info']['error_message'] = '响应为空'
                    print(f"❌ {ctx.display} 连接失败: 响应为空")
                    results[ctx.key] = False
            except Exception as e:
                ctx.web_data['ai_model_info']['status'] = 'error'
                ctx.web_data['ai_model_info']['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ctx.web_data['ai_model_info']['error_message'] = str(e)
                print(f"❌ {ctx.display} 连接失败: {e}")
                results[ctx.key] = False

        refresh_overview_from_context(ctx)

    if model_key:
        return results.get(model_key, False)
    return results


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data['price'] * 0.98,  # -2%
        "take_profit": price_data['price'] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }


def analyze_with_deepseek(symbol, price_data, config):
    """使用AI分析市场并生成交易信号（多交易对+动态杠杆+智能资金管理版本）"""

    # 获取账户余额信息
    try:
        balance = exchange.fetch_balance()
        available_balance = balance['USDT']['free']
        total_equity = balance['USDT']['total']
    except:
        available_balance = 1000  # 默认值
        total_equity = 1000

    # 🆕 智能资金管理：预计算所有可能的仓位组合
    current_price = price_data['price']
    max_usable_margin = available_balance * 0.8  # 最多使用80%余额作为保证金

    # 为不同信心等级设置仓位比例
    confidence_ratios = {
        'HIGH': 0.7,    # 高信心使用70%的可用保证金
        'MEDIUM': 0.5,  # 中信心使用50%
        'LOW': 0.3      # 低信心使用30%
    }

    # 预计算所有组合的仓位
    position_suggestions = {}
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs['contract_size']
    min_contracts = specs['min_contracts']
    min_quantity = get_symbol_min_amount(symbol)
    leverage_list = [config['leverage_min'], config['leverage_default'], config['leverage_max']]

    for confidence in ['HIGH', 'MEDIUM', 'LOW']:
        ratio = confidence_ratios[confidence]
        for lev in leverage_list:
            target_margin = max_usable_margin * ratio
            raw_quantity = (target_margin * lev / current_price) if current_price else 0
            base_quantity = max(raw_quantity, min_quantity)
            contracts = base_to_contracts(symbol, base_quantity)
            if min_contracts:
                contracts = max(contracts, min_contracts)
            adjusted_contracts = adjust_contract_quantity(symbol, contracts, round_up=True)
            adjusted_quantity = contracts_to_base(symbol, adjusted_contracts)
            adjusted_margin = adjusted_quantity * current_price / lev if lev else 0
            meets_min = adjusted_contracts >= (min_contracts if min_contracts else 0)
            meets_margin = adjusted_margin <= max_usable_margin if max_usable_margin else True
            key = f"{confidence}_{lev}"
            position_suggestions[key] = {
                'quantity': adjusted_quantity,
                'contracts': adjusted_contracts,
                'contract_size': contract_size,
                'value': adjusted_quantity * current_price,
                'margin': adjusted_margin,
                'meets_min': meets_min,
                'meets_margin': meets_margin,
                'meets': meets_min and meets_margin
            }

    can_trade = any(pos.get('meets') for pos in position_suggestions.values())
    position_suggestions['available_balance'] = available_balance
    position_suggestions['current_price'] = current_price
    position_suggestions['usable_margin'] = max_usable_margin
    position_suggestions['min_quantity'] = min_quantity
    position_suggestions['min_contracts'] = min_contracts
    position_suggestions['contract_size'] = contract_size

    ctx = get_active_context()

    if not can_trade:
        min_contracts_display = min_contracts if min_contracts else base_to_contracts(symbol, min_quantity)
        print(f"[{config['display']}] ⚠️ 余额不足：即使最大杠杆也无法满足最小交易量 {min_quantity} ({min_contracts_display:.3f} 张)")
        print(f"[{config['display']}] 💡 当前余额: {available_balance:.2f} USDT")
        print(f"[{config['display']}] 💡 建议充值至少: {(min_quantity * current_price / config['leverage_max']):.2f} USDT")

        fallback_signal = {
            "signal": "HOLD",
            "reason": f"账户余额不足({available_balance:.2f} USDT)，无法满足最小交易量要求({min_quantity}，约{min_contracts_display:.3f}张)，建议充值至少{(min_quantity * current_price / config['leverage_max']):.2f} USDT",
            "stop_loss": current_price * 0.98,
            "take_profit": current_price * 1.02,
            "confidence": "LOW",
            "leverage": config['leverage_default'],
            "order_quantity": 0,
            "is_insufficient_balance": True
        }

        fallback_signal['timestamp'] = price_data['timestamp']
        append_signal_record(symbol, fallback_signal, current_price, fallback_signal['timestamp'])
        ctx.metrics['signals_generated'] += 1

        print(f"[{config['display']}] 💡 跳过AI分析（余额不足），直接返回HOLD信号")
        return fallback_signal

    update_signal_validation(symbol, price_data['price'], price_data['timestamp'])

    token = symbol.split('/')[0] if '/' in symbol else symbol
    sentiment_text = ""
    sentiment_data = get_sentiment_indicators(token)

    if sentiment_data:
        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        sentiment_text = f"{token}市场情绪 乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
        print(f"[{config['display']}] {sentiment_text}")
    else:
        if token != 'BTC':
            print(f"[{config['display']}] ⚠️ {token}情绪数据不可用，尝试使用BTC市场情绪...")
            btc_sentiment = get_sentiment_indicators('BTC')
            if btc_sentiment:
                sign = '+' if btc_sentiment['net_sentiment'] >= 0 else ''
                sentiment_text = f"BTC市场情绪(参考) 乐观{btc_sentiment['positive_ratio']:.1%} 悲观{btc_sentiment['negative_ratio']:.1%} 净值{sign}{btc_sentiment['net_sentiment']:.3f}"
                print(f"[{config['display']}] {sentiment_text}")
            else:
                sentiment_text = "市场情绪暂无有效数据"
        else:
            sentiment_text = "市场情绪暂无有效数据"

    current_position = get_current_position(symbol)
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs['contract_size']
    min_contracts = max(specs['min_contracts'], base_to_contracts(symbol, get_symbol_min_amount(symbol)))
    min_contracts = adjust_contract_quantity(symbol, min_contracts, round_up=True) if min_contracts else 0
    min_quantity = contracts_to_base(symbol, min_contracts) if min_contracts else get_symbol_min_amount(symbol)
    ctx.metrics['ai_calls'] += 1

    prompt = build_professional_prompt(ctx, symbol, price_data, config, position_suggestions, sentiment_text, current_position)
    try:
        print(f"⏳ 正在调用{AI_PROVIDER.upper()} API ({AI_MODEL})...")
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system",
                 "content": f"""您是一位经验丰富的量化交易员，专注于加密货币{TRADE_CONFIG['timeframe']}周期的短线交易。
                核心原则:
                1. 数据驱动: 优先信任历史验证准确率 > 单次技术指标
                2. 风险优先: 宁可错过机会，不可冒险亏损
                3. 自我校准: 参考自身HIGH信心信号的历史表现
                4. 趋势为王: 不逆大趋势交易(20周期均线方向)

                决策流程:
                1. 先看历史准确率统计 → 评估当前状态可信度
                2. 再看技术指标趋势 → 确定市场方向
                3. 结合市场情绪 → 微调信心等级
                4. 评估持仓状态 → 决定开仓/平仓/观望
                5. 选择仓位建议表中的数量 → 禁止自行计算

                请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。
                 """},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1,
            timeout=30.0  # 30秒超时
        )
        print("✓ API调用成功")
        
        # 更新AI连接状态
        web_data['ai_model_info']['status'] = 'connected'
        web_data['ai_model_info']['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        web_data['ai_model_info']['error_message'] = None

        # 检查响应
        if not response or not response.choices:
            print(f"❌ {AI_PROVIDER.upper()}返回空响应")
            web_data['ai_model_info']['status'] = 'error'
            web_data['ai_model_info']['error_message'] = '响应为空'
            return create_fallback_signal(price_data)
        
        # 安全解析JSON
        result = response.choices[0].message.content
        if not result:
            print(f"❌ {AI_PROVIDER.upper()}返回空内容")
            return create_fallback_signal(price_data)
            
        print(f"\n{'='*60}")
        print(f"{AI_PROVIDER.upper()}原始回复:")
        print(result)
        print(f"{'='*60}\n")

        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)

            if signal_data is None:
                print("⚠️ JSON解析失败，使用备用信号")
                signal_data = create_fallback_signal(price_data)
            else:
                print(f"✓ 成功解析AI决策: {signal_data.get('signal')} - {signal_data.get('confidence')}")
        else:
            print("⚠️ 未找到JSON格式，使用备用信号")
            signal_data = create_fallback_signal(price_data)

        # 验证必需字段
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            missing = [f for f in required_fields if f not in signal_data]
            print(f"⚠️ 缺少必需字段: {missing}，使用备用信号")
            signal_data = create_fallback_signal(price_data)

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        record = append_signal_record(symbol, signal_data, price_data['price'], signal_data['timestamp'])
        history = signal_history[symbol]
        ctx.metrics['signals_generated'] += 1

        # 信号统计
        signal_count = len([s for s in history if s.get('signal') == record.get('signal')])
        total_signals = len(history)
        print(f"[{config['display']}] 信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)")

        # 信号连续性检查
        if len(history) >= 3:
            last_three = [s['signal'] for s in history[-3:]]
            if len(set(last_three)) == 1:
                print(f"[{config['display']}] ⚠️ 注意：连续3次{signal_data['signal']}信号")

        return signal_data

    except Exception as e:
        print(f"[{config['display']}] ❌ {AI_PROVIDER.upper()}分析失败: {e}")
        import traceback
        traceback.print_exc()
        ctx.metrics['ai_errors'] += 1
        # 更新AI连接状态
        web_data['ai_model_info']['status'] = 'error'
        web_data['ai_model_info']['last_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        web_data['ai_model_info']['error_message'] = str(e)
        fallback = create_fallback_signal(price_data)
        fallback['timestamp'] = price_data['timestamp']
        append_signal_record(symbol, fallback, price_data['price'], fallback['timestamp'])
        return fallback


def execute_trade(symbol, signal_data, price_data, config):
    """执行交易 - OKX版本（多交易对+动态杠杆+动态资金）"""
    global web_data

    current_position = get_current_position(symbol)

    # 🔴 紧急修复：防止频繁反转
    if current_position and signal_data['signal'] != 'HOLD':
        current_side = current_position['side']
        # 修正：正确处理HOLD情况
        if signal_data['signal'] == 'BUY':
            new_side = 'long'
        elif signal_data['signal'] == 'SELL':
            new_side = 'short'
        else:  # HOLD
            new_side = None

        # 如果只是方向反转，需要高信心才执行
        if new_side != current_side:
            if signal_data['confidence'] != 'HIGH':
                print(f"[{config['display']}] 🔒 非高信心反转信号，保持现有{current_side}仓")
                return

            # 检查最近信号历史，避免频繁反转
            if len(signal_history[symbol]) >= 2:
                last_signals = [s['signal'] for s in signal_history[symbol][-2:]]
                if signal_data['signal'] in last_signals:
                    print(f"[{config['display']}] 🔒 近期已出现{signal_data['signal']}信号，避免频繁反转")
                    return

    print(f"[{config['display']}] 交易信号: {signal_data.get('signal')}")
    print(f"[{config['display']}] 信心程度: {signal_data.get('confidence')}")
    print(f"[{config['display']}] 理由: {signal_data.get('reason')}")
    print(f"[{config['display']}] 止损: {format_currency(signal_data.get('stop_loss'))}")
    print(f"[{config['display']}] 止盈: {format_currency(signal_data.get('take_profit'))}")
    print(f"[{config['display']}] 当前持仓: {current_position}")

    if signal_data.get('signal', '').upper() == 'HOLD':
        print(f"[{config['display']}] ℹ️ HOLD 信号，不执行下单流程")
        return

    # 风险管理：低信心信号不执行
    if signal_data['confidence'] == 'LOW' and not config['test_mode']:
        print(f"[{config['display']}] ⚠️ 低信心信号，跳过执行")
        return

    if config['test_mode']:
        print(f"[{config['display']}] 测试模式 - 仅模拟交易")
        return

    try:
        # 🔒 获取全局执行锁，防止多个交易对并发下单导致保证金竞争
        with order_execution_lock:
            print(f"[{config['display']}] 🔒 已获取交易执行锁，开始处理...")

            # 📊 获取账户余额
            balance = exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            # 获取AI建议的杠杆和数量
            suggested_leverage = signal_data.get('leverage', config['leverage_default'])
            order_value = signal_data.get('order_value', 0)
            order_quantity = signal_data.get('order_quantity', 0)

            # 🆕 双重验证机制：智能计算实际可用保证金
            current_price = price_data['price']

            contract_specs = get_symbol_contract_specs(symbol)
            contract_size = contract_specs['contract_size']
            min_contracts = contract_specs.get('min_contracts') or 0
            if min_contracts and min_contracts > 0:
                min_contracts = adjust_contract_quantity(symbol, min_contracts, round_up=True)
            min_quantity = contracts_to_base(symbol, min_contracts) if min_contracts else get_symbol_min_amount(symbol)

            # 🔴 关键修复：从OKX balance结构中提取更准确的数据
            try:
                # 尝试从info.details中获取USDT的详细信息
                usdt_details = None
                if 'info' in balance and 'data' in balance['info']:
                    for data_item in balance['info']['data']:
                        if 'details' in data_item:
                            for detail in data_item['details']:
                                if detail.get('ccy') == 'USDT':
                                    usdt_details = detail
                                    break

                if usdt_details:
                    # 使用OKX的实际可用余额和保证金率计算
                    avail_bal = float(usdt_details.get('availBal', usdt_balance))
                    total_eq = float(usdt_details.get('eq', usdt_balance))
                    frozen_bal = float(usdt_details.get('frozenBal', 0))
                    current_imr = float(usdt_details.get('imr', 0))

                    print(f"[{config['display']}] 📊 OKX账户详情:")
                    print(f"[{config['display']}]    - 可用余额: {avail_bal:.2f} USDT")
                    print(f"[{config['display']}]    - 总权益: {total_eq:.2f} USDT")
                    print(f"[{config['display']}]    - 已冻结: {frozen_bal:.2f} USDT")
                    print(f"[{config['display']}]    - 已占用保证金: {current_imr:.2f} USDT")

                    # 🔴 方案B++：智能计算保证金（50%阈值 + 75%缓冲）
                    # 说明：考虑OKX隐藏buffer、手续费、价格波动等因素，使用更保守的参数
                    max_total_imr = total_eq * 0.50  # 总保证金不超过50%（应对OKX风控）
                    max_new_margin = max_total_imr - current_imr  # 可用于新仓位的保证金

                    # 取两者的较小值，并应用75%安全缓冲（应对价格波动、手续费、OKX buffer）
                    max_usable_margin = min(avail_bal, max_new_margin) * 0.75

                    print(f"[{config['display']}] 💡 智能计算:")
                    print(f"[{config['display']}]    - 最大允许总保证金: {max_total_imr:.2f} USDT (权益的50%)")
                    print(f"[{config['display']}]    - 可用于新仓位: {max_new_margin:.2f} USDT")
                    print(f"[{config['display']}]    - 最终可用保证金: {max_usable_margin:.2f} USDT (含75%安全缓冲)")
                else:
                    # 降级方案：简单计算
                    max_usable_margin = usdt_balance * 0.35
                    print(f"[{config['display']}] ⚠️ 未找到详细信息，使用简单计算: {max_usable_margin:.2f} USDT")
            except Exception as e:
                # 异常时使用保守策略
                max_usable_margin = usdt_balance * 0.35
                print(f"[{config['display']}] ⚠️ 解析balance失败: {e}，使用保守值: {max_usable_margin:.2f} USDT")

            # 为当前信心等级和杠杆计算有效仓位
            confidence = signal_data.get('confidence', 'MEDIUM')
            confidence_ratios = {'HIGH': 0.7, 'MEDIUM': 0.5, 'LOW': 0.3}
            ratio = confidence_ratios.get(confidence, 0.5)

            margin_pool = max_usable_margin * ratio
            expected_position_value = margin_pool * suggested_leverage
            expected_quantity = expected_position_value / current_price if current_price else 0
            expected_contracts = base_to_contracts(symbol, expected_quantity)
            expected_contracts = adjust_contract_quantity(symbol, max(expected_contracts, min_contracts), round_up=True) if current_price else min_contracts
            expected_quantity = contracts_to_base(symbol, expected_contracts)

            # 确定交易张数
            if order_quantity > 0:
                trade_contracts = base_to_contracts(symbol, order_quantity)
                trade_amount = contracts_to_base(symbol, trade_contracts)
                lower_bound = expected_quantity * 0.8
                upper_bound = expected_quantity * 1.2
                if expected_quantity > 0 and (trade_amount < lower_bound or trade_amount > upper_bound):
                    print(f"[{config['display']}] ⚠️ AI返回的数量 {trade_amount:.6f} 超出预期范围 [{lower_bound:.6f}, {upper_bound:.6f}]")
                    print(f"[{config['display']}] 🔧 自动调整为标准仓位: {expected_quantity:.6f}")
                    trade_contracts = expected_contracts
            elif order_value > 0:
                raw_quantity = order_value / current_price if current_price else 0
                trade_contracts = base_to_contracts(symbol, raw_quantity)
            else:
                trade_contracts = expected_contracts
                print(f"[{config['display']}] 💡 AI未指定数量，使用标准仓位: {contracts_to_base(symbol, trade_contracts):.6f}")

            if min_contracts and trade_contracts < min_contracts:
                print(f"[{config['display']}] ⚠️ 交易张数 {trade_contracts:.6f} 低于最小张数 {min_contracts:.6f}")
                test_margin = current_price * contracts_to_base(symbol, min_contracts) / suggested_leverage if current_price else 0
                if test_margin <= max_usable_margin:
                    print(f"[{config['display']}] 🔧 调整为最小交易量: {contracts_to_base(symbol, min_contracts):.6f}")
                    trade_contracts = min_contracts
                else:
                    print(f"[{config['display']}] ❌ 即使最小交易量也需要 {test_margin:.2f} USDT保证金，超出可用 {max_usable_margin:.2f} USDT")
                    print(f"[{config['display']}] 💡 建议充值至少: {(contracts_to_base(symbol, min_contracts) * current_price / suggested_leverage):.2f} USDT")
                    return

            trade_contracts = adjust_contract_quantity(symbol, max(trade_contracts, min_contracts), round_up=True)
            trade_amount = contracts_to_base(symbol, trade_contracts)

            if min_contracts and trade_contracts < min_contracts:
                print(f"[{config['display']}] ❌ 调整到交易精度后张数仍低于最小要求 {min_contracts}")
                return

            # 计算所需保证金（第1次验证）
            required_margin = current_price * trade_amount / suggested_leverage

            if required_margin > max_usable_margin:
                print(f"[{config['display']}] ⚠️ 初步验证：保证金不足")
                print(f"[{config['display']}] 需要: {required_margin:.2f} USDT")
                print(f"[{config['display']}] 可用: {max_usable_margin:.2f} USDT")

                # 🆕 尝试动态调整数量
                adjusted_contracts = base_to_contracts(symbol, (max_usable_margin * 0.95) * suggested_leverage / current_price if current_price else 0)
                adjusted_contracts = adjust_contract_quantity(symbol, max(adjusted_contracts, min_contracts), round_up=True)
                adjusted_amount = contracts_to_base(symbol, adjusted_contracts)
                if adjusted_contracts >= min_contracts and adjusted_amount >= min_quantity:
                    print(f"[{config['display']}] 💡 动态调整数量: {trade_amount:.6f} ({trade_contracts:.6f}张) → {adjusted_amount:.6f} ({adjusted_contracts:.6f}张)")
                    trade_contracts = adjusted_contracts
                    trade_amount = adjusted_amount
                    required_margin = current_price * trade_amount / suggested_leverage
                else:
                    print(f"[{config['display']}] ❌ 即使调整也无法满足最小交易量，跳过")
                    return

            # 显示初步计算结果
            print(f"[{config['display']}] 📊 初步计算参数:")
            print(f"[{config['display']}]    - 数量: {trade_amount:.6f} ({trade_contracts:.6f} 张, 合约面值 {contract_size:g})")
            print(f"[{config['display']}]    - 杠杆: {suggested_leverage}x")
            print(f"[{config['display']}]    - 所需保证金: {required_margin:.2f} USDT")
            print(f"[{config['display']}]    - 仓位价值: ${(current_price * trade_amount):.2f}")
            print(f"[{config['display']}]    - 保证金占用率: {(required_margin / max_usable_margin * 100):.1f}%")

            # ============ 🆕 关键改进：下单前实时验证 ============
            print(f"\n[{config['display']}] 🔄 下单前重新验证余额...")
            time.sleep(0.5)  # 短暂延迟，让其他线程订单生效

            # 📊 第2次余额获取（实时）+ 智能计算
            fresh_balance = exchange.fetch_balance()
            fresh_usdt = fresh_balance['USDT']['free']

            # 🔴 关键修复：应用同样的智能保证金计算
            try:
                # 解析OKX详细余额信息
                fresh_usdt_details = None
                if 'info' in fresh_balance and 'data' in fresh_balance['info']:
                    for data_item in fresh_balance['info']['data']:
                        if 'details' in data_item:
                            for detail in data_item['details']:
                                if detail.get('ccy') == 'USDT':
                                    fresh_usdt_details = detail
                                    break

                if fresh_usdt_details:
                    # 使用OKX的实际可用余额和保证金率计算
                    fresh_avail_bal = float(fresh_usdt_details.get('availBal', fresh_usdt))
                    fresh_total_eq = float(fresh_usdt_details.get('eq', fresh_usdt))
                    fresh_current_imr = float(fresh_usdt_details.get('imr', 0))

                    # 🔴 方案B++：智能计算保证金（50%阈值 + 75%缓冲）- 与第一阶段完全一致
                    # 说明：考虑OKX隐藏buffer、手续费、价格波动等因素，使用更保守的参数
                    fresh_max_total_imr = fresh_total_eq * 0.50  # 总保证金不超过50%（应对OKX风控）
                    fresh_max_new_margin = fresh_max_total_imr - fresh_current_imr

                    # 取两者的较小值，并应用75%安全缓冲（应对价格波动、手续费、OKX buffer）
                    fresh_max_margin = min(fresh_avail_bal, fresh_max_new_margin) * 0.75

                    print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                    print(f"[{config['display']}] 💡 实时智能计算:")
                    print(f"[{config['display']}]    - 总权益: {fresh_total_eq:.2f} USDT")
                    print(f"[{config['display']}]    - 已占用保证金: {fresh_current_imr:.2f} USDT")
                    print(f"[{config['display']}]    - 可用于新仓位: {fresh_max_new_margin:.2f} USDT")
                    print(f"[{config['display']}]    - 最终可用保证金: {fresh_max_margin:.2f} USDT (含75%安全缓冲)")
                else:
                    # 降级方案：简单计算
                    fresh_max_margin = fresh_usdt * 0.35
                    print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                    print(f"[{config['display']}] ⚠️ 未找到详细信息，使用简单计算: {fresh_max_margin:.2f} USDT")
            except Exception as e:
                # 异常时使用保守策略
                fresh_max_margin = fresh_usdt * 0.35
                print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                print(f"[{config['display']}] ⚠️ 实时解析失败: {e}，使用保守值: {fresh_max_margin:.2f} USDT")

            # 🆕 第2次验证
            if required_margin > fresh_max_margin:
                print(f"[{config['display']}] ❌ 实时验证失败：保证金不足")
                print(f"[{config['display']}] 需要: {required_margin:.2f} USDT")
                print(f"[{config['display']}] 实时: {fresh_max_margin:.2f} USDT")
                print(f"[{config['display']}] 💡 可能其他交易对已占用保证金")

                # 🆕 再次尝试动态调整
                final_adjusted_contracts = base_to_contracts(symbol, (fresh_max_margin * 0.95) * suggested_leverage / current_price if current_price else 0)
                final_adjusted_contracts = adjust_contract_quantity(symbol, max(final_adjusted_contracts, min_contracts), round_up=True)
                final_adjusted_amount = contracts_to_base(symbol, final_adjusted_contracts)
                if final_adjusted_contracts >= min_contracts and final_adjusted_amount >= min_quantity:
                    print(f"[{config['display']}] 💡 最终调整数量: {trade_amount:.6f} ({trade_contracts:.6f}张) → {final_adjusted_amount:.6f} ({final_adjusted_contracts:.6f}张)")
                    trade_contracts = final_adjusted_contracts
                    trade_amount = final_adjusted_amount
                    required_margin = current_price * trade_amount / suggested_leverage
                else:
                    print(f"[{config['display']}] ❌ 无法调整，彻底放弃")
                    return

            print(f"[{config['display']}] ✅ 实时验证通过")
            print(f"[{config['display']}] 📊 最终交易参数:")
            print(f"[{config['display']}]    - 数量: {trade_amount:.6f} ({trade_contracts:.6f} 张)")
            print(f"[{config['display']}]    - 杠杆: {suggested_leverage}x")
            print(f"[{config['display']}]    - 所需保证金: {required_margin:.2f} USDT")

            # ========== P0-2: 全局风险管理检查 ==========
            trade_value = current_price * trade_amount
            print(f"\n[{config['display']}] 🛡️ 全局风险管理检查...")
            if not risk_manager.can_open_position(exchange, symbol, trade_value, suggested_leverage):
                print(f"[{config['display']}] ❌ 全局风险管理拒绝开仓")
                return
            
            # 🆕 在验证通过后才设置杠杆（避免验证失败导致的杠杆副作用）
            current_leverage = current_position['leverage'] if current_position else config['leverage_default']
            if suggested_leverage != current_leverage:
                try:
                    exchange.set_leverage(
                        suggested_leverage,
                        symbol,
                        {'mgnMode': 'cross'}
                    )
                    print(f"[{config['display']}] ✓ 杠杆已设置为 {suggested_leverage}x")
                except Exception as e:
                    print(f"[{config['display']}] ⚠️ 杠杆设置失败: {e}")
                    # 如果杠杆设置失败，使用当前杠杆重新计算
                    suggested_leverage = current_leverage
                    required_margin = current_price * trade_amount / suggested_leverage
                    print(f"[{config['display']}] 使用当前杠杆 {suggested_leverage}x")

            # ============ 🆕 执行交易（带重试机制） ============
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"\n[{config['display']}] 📤 执行交易（尝试 {attempt + 1}/{max_retries}）...")

                    # 执行交易逻辑 - tag是经纪商api
                    if signal_data['signal'] == 'BUY':
                        if current_position and current_position['side'] == 'short':
                            close_contracts = float(current_position.get('size', 0) or 0)
                            base_token = symbol.split('/')[0]
                            close_amount = contracts_to_base(symbol, close_contracts)
                            print(f"[{config['display']}] 平空仓并开多仓... 平空 {close_contracts:.6f} 张 (~{close_amount:.6f} {base_token})")
                            # 平空仓
                            exchange.create_market_order(
                                symbol, 'buy', close_contracts,
                                params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE'}
                            )
                            time.sleep(1)
                            # 开多仓
                            exchange.create_market_order(
                                symbol, 'buy', trade_contracts,
                                params={'tag': '60bb4a8d3416BCDE'}
                            )
                        elif current_position and current_position['side'] == 'long':
                            print(f"[{config['display']}] 已有多头持仓，保持现状")
                            return  # 不记录重复交易
                        else:
                            # 无持仓时开多仓
                            print(f"[{config['display']}] 开多仓...")
                            exchange.create_market_order(
                                symbol, 'buy', trade_contracts,
                                params={'tag': '60bb4a8d3416BCDE'}
                            )

                    elif signal_data['signal'] == 'SELL':
                        if current_position and current_position['side'] == 'long':
                            close_contracts = float(current_position.get('size', 0) or 0)
                            base_token = symbol.split('/')[0]
                            close_amount = contracts_to_base(symbol, close_contracts)
                            print(f"[{config['display']}] 平多仓并开空仓... 平多 {close_contracts:.6f} 张 (~{close_amount:.6f} {base_token})")
                            # 平多仓
                            exchange.create_market_order(
                                symbol, 'sell', close_contracts,
                                params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE'}
                            )
                            time.sleep(1)
                            # 开空仓
                            exchange.create_market_order(
                                symbol, 'sell', trade_contracts,
                                params={'tag': '60bb4a8d3416BCDE'}
                            )
                        elif current_position and current_position['side'] == 'short':
                            print(f"[{config['display']}] 已有空头持仓，保持现状")
                            return  # 不记录重复交易
                        else:
                            # 无持仓时开空仓
                            print(f"[{config['display']}] 开空仓...")
                            exchange.create_market_order(
                                symbol, 'sell', trade_contracts,
                                params={'tag': '60bb4a8d3416BCDE'}
                            )

                    print(f"[{config['display']}] ✓ 订单执行成功")
                    break  # 成功则跳出重试循环

                except ccxt.InsufficientFunds as e:
                    # 🆕 捕获51008保证金不足错误
                    print(f"[{config['display']}] ❌ 保证金不足错误: {e}")

                    if attempt < max_retries - 1:
                        # 还有重试机会，尝试减少50%数量
                        print(f"[{config['display']}] 💡 尝试减少50%数量重试...")
                        trade_contracts = adjust_contract_quantity(symbol, trade_contracts * 0.5, round_up=True)
                        trade_amount = contracts_to_base(symbol, trade_contracts)
                        if min_contracts and trade_contracts < min_contracts:
                            print(f"[{config['display']}] ❌ 减少后仍低于最小张数{min_contracts}，放弃")
                            return
                        required_margin = current_price * trade_amount / suggested_leverage
                        print(f"[{config['display']}] 新数量: {trade_amount:.6f} ({trade_contracts:.6f}张), 新保证金: {required_margin:.2f} USDT")
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        print(f"[{config['display']}] ❌ 重试次数已用完，彻底放弃")
                        return

                except Exception as e:
                    print(f"[{config['display']}] ❌ 订单执行失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"[{config['display']}] 等待2秒后重试...")
                        time.sleep(2)
                    else:
                        import traceback
                        traceback.print_exc()
                        return

            # 等待订单完全生效
            time.sleep(2)

            # 更新持仓信息
            updated_position = get_current_position(symbol)
            print(f"[{config['display']}] 更新后持仓: {updated_position}")
            
            # 🆕 验证持仓变化是否符合预期
            expected_side = 'long' if signal_data['signal'] == 'BUY' else 'short'
            if not updated_position:
                print(f"[{config['display']}] ⚠️ 警告：订单执行后无持仓，可能执行失败")
                return  # 不记录失败的交易
            elif updated_position.get('side') != expected_side:
                print(f"[{config['display']}] ⚠️ 警告：持仓方向({updated_position.get('side')})与预期({expected_side})不符")
                return  # 不记录异常交易
            
            # ✅ 验证通过，记录交易成功
            print(f"[{config['display']}] ✅ 订单验证通过，持仓方向符合预期: {updated_position.get('side')}")
            
            ctx = get_active_context()
            # 根据持仓变化更新统计（在验证通过后）
            if current_position and current_position.get('side') != updated_position.get('side'):
                # 方向发生变化，算一次平仓+开仓
                ctx.metrics['trades_closed'] += 1
                ctx.metrics['trades_opened'] += 1
            elif not current_position and updated_position:
                # 从无持仓到有持仓
                ctx.metrics['trades_opened'] += 1

            # 记录交易历史（使用线程锁保护）
            trade_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': signal_data['signal'],
                'side': 'buy' if signal_data['signal'] == 'BUY' else 'sell',  # 🆕 添加 side 字段供前端显示
                'price': price_data['price'],
                'amount': trade_amount,
                'contracts': trade_contracts,
                'leverage': suggested_leverage,
                'confidence': signal_data['confidence'],
                'reason': signal_data['reason']
            }

            with data_lock:
                web_data['symbols'][symbol]['trade_history'].append(trade_record)
                if len(web_data['symbols'][symbol]['trade_history']) > 100:  # 只保留最近100条
                    web_data['symbols'][symbol]['trade_history'].pop(0)

                # 更新持仓信息
                web_data['symbols'][symbol]['current_position'] = updated_position

                # 更新杠杆记录
                web_data['symbols'][symbol]['performance']['current_leverage'] = suggested_leverage
                web_data['symbols'][symbol]['performance']['suggested_leverage'] = suggested_leverage
                web_data['symbols'][symbol]['performance']['last_order_value'] = price_data['price'] * trade_amount
                web_data['symbols'][symbol]['performance']['last_order_quantity'] = trade_amount
                web_data['symbols'][symbol]['performance']['last_order_contracts'] = trade_contracts

            print(f"[{config['display']}] 🔓 释放交易执行锁")
            # with块结束，自动释放order_execution_lock

    except Exception as e:
        print(f"[{config['display']}] ❌ 订单执行失败: {e}")
        import traceback
        traceback.print_exc()


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            print(f"第{attempt + 1}次尝试失败，进行重试...")
            time.sleep(2)

        except Exception as e:
            print(f"第{attempt + 1}次尝试异常: {e}")
            import traceback
            traceback.print_exc()
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(2)

    return create_fallback_signal(price_data)


def wait_for_next_period():
    """等待到下一个5分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（每5分钟：00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55分钟）
    INTERVAL_MINUTES = int(os.getenv('TRADE_INTERVAL_MINUTES', '5'))
    if INTERVAL_MINUTES <= 0:
        INTERVAL_MINUTES = 5
    next_period_minute = ((current_minute // INTERVAL_MINUTES) + 1) * INTERVAL_MINUTES
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait


def trading_bot():
    # 等待到整点再执行
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    """主交易机器人函数"""
    global web_data, initial_balance
    
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        return

    print(f"BTC当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        print("⚠️ 使用备用交易信号")

    # 3. 更新Web数据
    try:
        balance = exchange.fetch_balance()
        current_equity = balance['USDT']['total']
        
        # 设置初始余额
        if initial_balance is None:
            initial_balance = current_equity
        
        web_data['account_info'] = {
            'usdt_balance': balance['USDT']['free'],
            'total_equity': current_equity
        }
        
        # 记录收益曲线数据
        current_position = get_current_position()
        unrealized_pnl = current_position.get('unrealized_pnl', 0) if current_position else 0
        total_profit = current_equity - initial_balance
        profit_rate = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        profit_point = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'equity': current_equity,
            'profit': total_profit,
            'profit_rate': profit_rate,
            'unrealized_pnl': unrealized_pnl
        }
        web_data['profit_curve'].append(profit_point)
        
        # 只保留最近200个数据点（约50小时）
        if len(web_data['profit_curve']) > 200:
            web_data['profit_curve'].pop(0)
            
    except Exception as e:
        print(f"更新余额失败: {e}")
    
    web_data['current_price'] = price_data['price']
    web_data['current_position'] = get_current_position()
    web_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存K线数据
    web_data['kline_data'] = price_data['kline_data']
    
    # 保存AI决策
    ai_decision = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'signal': signal_data['signal'],
        'confidence': signal_data['confidence'],
        'reason': signal_data['reason'],
        'stop_loss': signal_data.get('stop_loss', 0),
        'take_profit': signal_data.get('take_profit', 0),
        'price': price_data['price']
    }
    web_data['ai_decisions'].append(ai_decision)
    if len(web_data['ai_decisions']) > 50:  # 只保留最近50条
        web_data['ai_decisions'].pop(0)
    
    # 更新性能统计
    if web_data['current_position']:
        web_data['performance']['total_profit'] = web_data['current_position'].get('unrealized_pnl', 0)

    # 4. 执行交易
    execute_trade(signal_data, price_data)


def run_symbol_cycle(symbol, config):
    """单个交易对的完整执行周期"""
    ctx = get_active_context()
    try:
        ensure_symbol_state(symbol)

        print(f"\n[{config['display']}] {'='*50}")
        print(f"[{config['display']}] 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 获取K线数据
        price_data = get_symbol_ohlcv_enhanced(symbol, config)
        if not price_data:
            print(f"[{config['display']}] ❌ 获取数据失败，跳过")
            return

        price_decimals = get_price_decimals(symbol)
        print(f"[{config['display']}] 当前价格: ${price_data['price']:,.{price_decimals}f} ({price_data['price_change']:+.2f}%)")
        
        # ========== P0-1: 优先检查止损止盈（最高优先级） ==========
        current_price = price_data['price']
        if check_stop_loss_take_profit(symbol, current_price, config):
            print(f"[{config['display']}] ✓ 止损/止盈已触发并执行，跳过本轮AI分析")
            # 更新Web数据后直接返回
            with data_lock:
                web_data['symbols'][symbol].update({
                    'current_price': current_price,
                    'kline_data': price_data['kline_data'],
                    'current_position': get_current_position(symbol),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            return
        
        # ========== P0-3: 检测异常市场 ==========
        is_abnormal, abnormal_reason = detect_abnormal_market(symbol, price_data, config)
        if is_abnormal:
            print(f"[{config['display']}] ⚠️ 检测到异常市场: {abnormal_reason}")
            print(f"[{config['display']}] 🔒 进入保护模式，暂停交易")
            
            # 异常行情下优先保护现有持仓（再次检查止损）
            check_stop_loss_take_profit(symbol, current_price, config)
            
            # 更新Web数据后直接返回，跳过AI分析和新交易
            with data_lock:
                web_data['symbols'][symbol].update({
                    'current_price': current_price,
                    'kline_data': price_data['kline_data'],
                    'current_position': get_current_position(symbol),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                web_data['symbols'][symbol]['performance']['price_change'] = price_data.get('price_change', 0)
            return

        # 2. AI分析
        signal_data = analyze_with_deepseek(symbol, price_data, config)

        # 3. 更新Web数据
        with data_lock:
            web_data['symbols'][symbol].update({
                'current_price': price_data['price'],
                'kline_data': price_data['kline_data'],
                'current_position': get_current_position(symbol),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 更新性能数据中的价格变化
            web_data['symbols'][symbol]['performance']['price_change'] = price_data.get('price_change', 0)

            # 保存AI决策
            ai_decision = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': signal_data['signal'],
                'confidence': signal_data['confidence'],
                'reason': signal_data['reason'],
                'stop_loss': signal_data.get('stop_loss', 0),
                'take_profit': signal_data.get('take_profit', 0),
                'leverage': signal_data.get('leverage', config['leverage_default']),
                'order_value': signal_data.get('order_value', 0),
                'order_quantity': signal_data.get('order_quantity', 0),
                'price': price_data['price']
            }
            web_data['symbols'][symbol]['ai_decisions'].append(ai_decision)
            if len(web_data['symbols'][symbol]['ai_decisions']) > 50:
                web_data['symbols'][symbol]['ai_decisions'].pop(0)

        # 4. 执行交易
        execute_trade(symbol, signal_data, price_data, config)

        print(f"[{config['display']}] ✓ 周期完成")

    except Exception as e:
        print(f"[{config.get('display', symbol)}] ❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


def run_all_symbols_parallel(model_display: str):
    """并行执行所有交易对（针对单个模型上下文）"""
    print("\n" + "="*70)
    print(f"🚀 [{model_display}] 开始新一轮分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=len(TRADE_CONFIGS)) as executor:
        futures = []
        for symbol, config in TRADE_CONFIGS.items():
            future = executor.submit(run_symbol_cycle, symbol, config)
            futures.append((symbol, future))

            # 添加延迟避免API限频
            time.sleep(2)

        # 等待所有任务完成
        for symbol, future in futures:
            try:
                future.result(timeout=60)  # 60秒超时
            except Exception as e:
                print(f"[{model_display} | {TRADE_CONFIGS[symbol]['display']}] ⚠️ 任务异常: {e}")

    print("\n" + "="*70)
    print(f"✓ [{model_display}] 本轮分析完成")
    print("="*70 + "\n")


def main():
    """主入口：同时调度多模型、多交易对"""
    print("\n" + "="*70)
    print("🤖 多交易对自动交易机器人启动")
    print("="*70)
    print(f"启用模型: {', '.join([MODEL_CONTEXTS[key].display for key in MODEL_ORDER])}")
    print(f"交易对数量: {len(TRADE_CONFIGS)}")
    print(f"交易对列表: {', '.join([c['display'] for c in TRADE_CONFIGS.values()])}")
    print("="*70 + "\n")

    test_mode_count = sum(1 for c in TRADE_CONFIGS.values() if c.get('test_mode', True))
    if test_mode_count > 0:
        print(f"⚠️  {test_mode_count}/{len(TRADE_CONFIGS)} 个交易对处于测试模式")
    else:
        print("🔴 实盘交易模式 - 请谨慎操作！")

    print("\n初始化各模型的 OKX 账户...")
    for model_key in MODEL_ORDER:
        ctx = MODEL_CONTEXTS[model_key]
        sub_account = getattr(ctx, 'sub_account', None) or '主账户'
        print(f"\n[{ctx.display}] 绑定子账户: {sub_account}")
        with activate_context(ctx):
            if not setup_exchange():
                print(f"❌ {ctx.display} 交易所初始化失败，程序退出")
                return
            capture_balance_snapshot(ctx)
            refresh_overview_from_context(ctx)
        print(f"✓ {ctx.display} 交易所配置完成")

    print("\n系统参数：")
    print(f"- 执行模式: 每模型并行交易对")
    print(f"- 执行频率: 每5分钟整点 (00,05,10,15,20,25,30,35,40,45,50,55)")
    print(f"- API防限频延迟: 2秒/交易对\n")

    record_overview_point(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    while True:
        wait_seconds = wait_for_next_period()
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        cycle_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for model_key in MODEL_ORDER:
            ctx = MODEL_CONTEXTS[model_key]
            with activate_context(ctx):
                run_all_symbols_parallel(ctx.display)
                capture_balance_snapshot(ctx, cycle_timestamp)
                refresh_overview_from_context(ctx)

        record_overview_point(cycle_timestamp)
        history_store.compress_if_needed(datetime.now())
        time.sleep(60)


if __name__ == "__main__":
    main()
def get_active_context() -> ModelContext:
    if ACTIVE_CONTEXT is None:
        raise RuntimeError("当前没有激活的模型上下文。")
    return ACTIVE_CONTEXT
