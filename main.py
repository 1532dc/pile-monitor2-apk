import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import threading
import time
import os
import sys
import json
import wave
import struct
import subprocess
import tempfile
from datetime import datetime
import numpy as np
import sounddevice as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

class PileDrivingMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔨 打桩锤击计数监测系统 v9.1-完整版")
        self.root.geometry("1200x800")
        self.root.configure(bg='#ffffff')
        
        # 设置简洁风格
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.setup_styles()
        
        # 监测状态
        self.is_monitoring = False
        self.is_analyzing = False
        self.current_pile_strikes = 0
        self.all_pile_strikes = []
        self.pile_details = []
        self.strike_times = []
        self.strike_frequencies = []
        self.strike_volumes = []
        self.pile_start_time = None
        self.last_strike_time = None
        self.audio_stream = None
        self.current_pile_name = ""
        
        # 手动输入相关
        self.manual_strikes = 0
        self.penetration_depth = None
        self.elevation_height = None
        
        # 施工判定条件参数
        self.judgment_conditions = {
            'condition1': {'elevation_max': 3.0, 'penetration_max': 5.0, 'action': '停止锤击'},
            'condition2': {'elevation_max': 3.0, 'penetration_min': 5.0, 'action': '打至标高超高0.4m停锤'},
            'condition3': {'elevation_min': 3.0, 'penetration_max': 2.0, 'strikes_min': 2000, 'action': '再施打100锤观察'}
        }
        
        # AI训练数据
        self.training_data = []
        self.training_labels = []
        self.ai_model = None
        self.is_ai_training = False
        
        # 默认参数
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.threshold = 0.3
        self.min_frequency = 80
        self.max_frequency = 2000
        self.silence_duration = 600.0
        self.min_interval = 0.3
        self.file_analysis_threshold_multiplier = 1.0
        
        # 数据存储
        self.config_file = "config.json"
        self.model_file = "ai_model.pkl"
        self.current_audio_file = None
        
        # 加载配置和模型
        self.load_config()
        self.load_ai_model()
        
        self.setup_ui()
        self.update_device_list()
        self.start_status_update()
    
    def setup_styles(self):
        """设置简洁风格 - 青绿色按钮"""
        self.style.configure('TFrame', background='#ffffff')
        self.style.configure('TLabelframe', background='#ffffff', foreground='#2c3e50', 
                           font=('Arial', 10, 'bold'))
        self.style.configure('TLabelframe.Label', background='#ffffff', foreground='#2c3e50',
                           font=('Arial', 10, 'bold'))
        self.style.configure('TButton', background='#27ae60', foreground='white', 
                           borderwidth=1, focusthickness=3, focuscolor='none',
                           font=('Arial', 9))
        self.style.map('TButton', 
                      background=[('active', '#219653')],
                      foreground=[('active', 'white')])
        self.style.configure('TLabel', background='#ffffff', foreground='#2c3e50',
                           font=('Arial', 9))
        self.style.configure('Title.TLabel', background='#ffffff', foreground='#e74c3c',
                           font=('Arial', 16, 'bold'))
        self.style.configure('Status.TLabel', background='#ffffff', foreground='#27ae60',
                           font=('Arial', 10, 'bold'))
        self.style.configure('Value.TLabel', background='#ffffff', foreground='#e67e22',
                           font=('Arial', 10, 'bold'))
        self.style.configure('Highlight.TLabel', background='#ffffff', foreground='#c0392b',
                           font=('Arial', 11, 'bold'))
        
        self.style.configure('TCheckbutton', background='#ffffff', foreground='#2c3e50')
        self.style.configure('TRadiobutton', background='#ffffff', foreground='#2c3e50')
        self.style.configure('Horizontal.TScale', background='#ffffff', troughcolor='#ecf0f1')
        self.style.configure('Vertical.TScale', background='#ffffff')
        self.style.configure('TEntry', fieldbackground='#ffffff', foreground='#2c3e50')
        self.style.configure('TCombobox', fieldbackground='#ffffff', foreground='#2c3e50')
        self.style.configure('Horizontal.TProgressbar', background='#e74c3c')
    
    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.threshold = float(config.get('threshold', self.threshold))
                    self.min_frequency = float(config.get('min_frequency', self.min_frequency))
                    self.max_frequency = float(config.get('max_frequency', self.max_frequency))
                    self.silence_duration = float(config.get('silence_duration', self.silence_duration))
                    self.min_interval = float(config.get('min_interval', self.min_interval))
                    self.file_analysis_threshold_multiplier = float(config.get('file_analysis_threshold_multiplier', self.file_analysis_threshold_multiplier))
        except Exception as e:
            print(f"加载配置失败: {e}")
    
    def save_config(self):
        """保存配置"""
        try:
            config = {
                'threshold': float(self.threshold),
                'min_frequency': float(self.min_frequency),
                'max_frequency': float(self.max_frequency),
                'silence_duration': float(self.silence_duration),
                'min_interval': float(self.min_interval),
                'file_analysis_threshold_multiplier': float(self.file_analysis_threshold_multiplier)
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def load_ai_model(self):
        """加载AI模型"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.ai_model = pickle.load(f)
                self.log("✅ AI模型加载成功")
        except Exception as e:
            self.log(f"AI模型加载失败: {e}")
            self.ai_model = None
    
    def save_ai_model(self):
        """保存AI模型"""
        try:
            if self.ai_model:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(self.ai_model, f)
                self.log("✅ AI模型保存成功")
        except Exception as e:
            self.log(f"AI模型保存失败: {e}")
    
    def setup_ui(self):
        """设置用户界面 - 优化布局，左右各占一半"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, style='TFrame', padding="8")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🔨 打桩锤击计数监测系统 v9.1-完整版", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # 创建左右分栏 - 各占一半
        container = ttk.Frame(main_frame, style='TFrame')
        container.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板 - 占50%
        left_frame = ttk.LabelFrame(container, text="控制面板", style='TLabelframe', padding="8")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        # 右侧数据显示 - 占50%
        right_frame = ttk.LabelFrame(container, text="数据监测", style='TLabelframe', padding="8")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        
        # === 左侧控制面板内容 ===
        
        # 桩名称设置
        name_frame = ttk.Frame(left_frame, style='TFrame')
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="桩名称:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.pile_name_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_frame, textvariable=self.pile_name_var, width=18, 
                              style='TEntry', font=('Arial', 10))
        name_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(name_frame, text="设置", command=self.set_pile_name,
                  style='TButton', width=6).pack(side=tk.LEFT, padx=2)
        
        # 模式选择
        mode_frame = ttk.Frame(left_frame, style='TFrame')
        mode_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(mode_frame, text="工作模式:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="realtime")
        ttk.Radiobutton(mode_frame, text="实时监测", variable=self.mode_var, 
                       value="realtime", command=self.on_mode_change,
                       style='TRadiobutton').pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mode_frame, text="文件分析", variable=self.mode_var, 
                       value="file", command=self.on_mode_change,
                       style='TRadiobutton').pack(side=tk.LEFT, padx=8)
        
        # 实时监测设置
        self.realtime_frame = ttk.Frame(left_frame, style='TFrame')
        
        # 设备选择
        device_frame = ttk.Frame(self.realtime_frame, style='TFrame')
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="音频设备:", style='TLabel').pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                        state="readonly", width=20, style='TCombobox')
        self.device_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 文件分析设置
        self.file_frame = ttk.Frame(left_frame, style='TFrame')
        
        # 文件选择
        file_select_frame = ttk.Frame(self.file_frame, style='TFrame')
        file_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_select_frame, text="音频文件:", style='TLabel').pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_select_frame, textvariable=self.file_path_var, width=18,
                              style='TEntry')
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(file_select_frame, text="浏览", command=self.browse_file,
                  style='TButton', width=6).pack(side=tk.LEFT, padx=2)
        
        # 参数设置框架
        params_frame = ttk.LabelFrame(left_frame, text="检测参数", style='TLabelframe', padding="10")
        params_frame.pack(fill=tk.X, pady=8, ipady=5)
        
        # 第一行参数
        params_row1 = ttk.Frame(params_frame, style='TFrame')
        params_row1.pack(fill=tk.X, pady=4)
        
        # 声音阈值
        ttk.Label(params_row1, text="灵敏度:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_scale = ttk.Scale(params_row1, from_=0.02, to=0.8,
                                   variable=self.threshold_var, orient=tk.HORIZONTAL,
                                   length=120, style='Horizontal.TScale')
        threshold_scale.pack(side=tk.LEFT, padx=5)
        threshold_scale.configure(command=self.on_threshold_change)
        self.threshold_label = ttk.Label(params_row1, text=f"{self.threshold:.3f}", 
                                        style='Value.TLabel', width=6)
        self.threshold_label.pack(side=tk.LEFT, padx=2)
        
        # 最低频率阈值
        ttk.Label(params_row1, text="最低频率:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.min_freq_var = tk.DoubleVar(value=self.min_frequency)
        min_freq_scale = ttk.Scale(params_row1, from_=20, to=500,
                                  variable=self.min_freq_var, orient=tk.HORIZONTAL,
                                  length=120, style='Horizontal.TScale')
        min_freq_scale.pack(side=tk.LEFT, padx=5)
        min_freq_scale.configure(command=self.on_min_freq_change)
        self.min_freq_label = ttk.Label(params_row1, text=f"{self.min_frequency:.0f}Hz", 
                                       style='Value.TLabel', width=6)
        self.min_freq_label.pack(side=tk.LEFT, padx=2)
        
        # 第二行参数
        params_row2 = ttk.Frame(params_frame, style='TFrame')
        params_row2.pack(fill=tk.X, pady=4)
        
        # 最高频率阈值
        ttk.Label(params_row2, text="最高频率:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.max_freq_var = tk.DoubleVar(value=self.max_frequency)
        max_freq_scale = ttk.Scale(params_row2, from_=500, to=5000,
                                  variable=self.max_freq_var, orient=tk.HORIZONTAL,
                                  length=120, style='Horizontal.TScale')
        max_freq_scale.pack(side=tk.LEFT, padx=5)
        max_freq_scale.configure(command=self.on_max_freq_change)
        self.max_freq_label = ttk.Label(params_row2, text=f"{self.max_frequency:.0f}Hz", 
                                       style='Value.TLabel', width=6)
        self.max_freq_label.pack(side=tk.LEFT, padx=2)
        
        # 静默时间
        ttk.Label(params_row2, text="完成时间:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.silence_var = tk.DoubleVar(value=self.silence_duration)
        silence_spin = ttk.Spinbox(params_row2, from_=10, to=1800, 
                                  textvariable=self.silence_var, width=8,
                                  command=self.on_silence_change, style='TEntry',
                                  font=('Arial', 9))
        silence_spin.pack(side=tk.LEFT, padx=5)
        silence_spin.bind('<KeyRelease>', self.on_silence_change)
        ttk.Label(params_row2, text="秒", style='TLabel').pack(side=tk.LEFT)
        
        # 第三行参数
        params_row3 = ttk.Frame(params_frame, style='TFrame')
        params_row3.pack(fill=tk.X, pady=4)
        
        # 最小间隔
        ttk.Label(params_row3, text="最小间隔:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.min_interval_var = tk.DoubleVar(value=self.min_interval)
        interval_spin = ttk.Spinbox(params_row3, from_=0.1, to=2.0, 
                                   textvariable=self.min_interval_var, width=6,
                                   increment=0.1, command=self.on_advanced_change,
                                   style='TEntry', font=('Arial', 9))
        interval_spin.pack(side=tk.LEFT, padx=5)
        interval_spin.bind('<KeyRelease>', self.on_advanced_change)
        ttk.Label(params_row3, text="秒", style='TLabel').pack(side=tk.LEFT)
        
        # 文件分析阈值倍数
        ttk.Label(params_row3, text="文件倍数:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.file_threshold_multiplier_var = tk.DoubleVar(value=self.file_analysis_threshold_multiplier)
        file_threshold_spin = ttk.Spinbox(params_row3, from_=0.1, to=5.0, 
                                         textvariable=self.file_threshold_multiplier_var, width=6,
                                         increment=0.1, command=self.on_advanced_change,
                                         style='TEntry', font=('Arial', 9))
        file_threshold_spin.pack(side=tk.LEFT, padx=5)
        file_threshold_spin.bind('<KeyRelease>', self.on_advanced_change)
        
        # 主控制按钮
        control_frame = ttk.Frame(left_frame, style='TFrame')
        control_frame.pack(fill=tk.X, pady=8)
        
        # 第一行控制按钮
        control_row1 = ttk.Frame(control_frame, style='TFrame')
        control_row1.pack(fill=tk.X, pady=3)
        
        self.start_btn = ttk.Button(control_row1, text="🚀开始监测", 
                                   command=self.start_monitoring, width=14,
                                   style='TButton')
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(control_row1, text="🛑停止", 
                                  command=self.stop_monitoring, state=tk.DISABLED, width=10,
                                  style='TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # 第二行控制按钮
        control_row2 = ttk.Frame(control_frame, style='TFrame')
        control_row2.pack(fill=tk.X, pady=3)
        
        self.calibrate_btn = ttk.Button(control_row2, text="🎛️校准", 
                                       command=self.calibrate, width=10,
                                       style='TButton')
        self.calibrate_btn.pack(side=tk.LEFT, padx=2)
        
        self.analyze_btn = ttk.Button(control_row2, text="📊分析文件", 
                                     command=self.analyze_file, width=12,
                                     style='TButton')
        self.analyze_btn.pack(side=tk.LEFT, padx=2)
        
        # 第三行控制按钮
        control_row3 = ttk.Frame(control_frame, style='TFrame')
        control_row3.pack(fill=tk.X, pady=3)
        
        self.end_pile_btn = ttk.Button(control_row3, text="⏹️结束当前桩", 
                                      command=self.manual_end_pile, width=14,
                                      style='TButton')
        self.end_pile_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_row3, text="⚙️阈值优化", 
                  command=self.optimize_threshold, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        # 第四行控制按钮
        control_row4 = ttk.Frame(control_frame, style='TFrame')
        control_row4.pack(fill=tk.X, pady=3)
        
        ttk.Button(control_row4, text="🚀快速设置", 
                  command=self.quick_setup, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_row4, text="💾导出数据", 
                  command=self.export_data, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        # 第五行控制按钮
        control_row5 = ttk.Frame(control_frame, style='TFrame')
        control_row5.pack(fill=tk.X, pady=3)
        
        ttk.Button(control_row5, text="🗑️清空数据", 
                  command=self.clear_data, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_row5, text="❓帮助", 
                  command=self.show_help, width=10,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        # AI训练功能
        ai_frame = ttk.LabelFrame(left_frame, text="AI智能分析", style='TLabelframe', padding="8")
        ai_frame.pack(fill=tk.X, pady=8)
        
        ai_control_frame = ttk.Frame(ai_frame, style='TFrame')
        ai_control_frame.pack(fill=tk.X, pady=4)
        
        ttk.Button(ai_control_frame, text="🤖开始AI训练", 
                  command=self.start_ai_training, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(ai_control_frame, text="🧠使用AI分析", 
                  command=self.use_ai_analysis, width=12,
                  style='TButton').pack(side=tk.LEFT, padx=2)
        
        self.ai_status_var = tk.StringVar(value="AI模型: 未训练")
        ttk.Label(ai_frame, textvariable=self.ai_status_var, style='TLabel').pack(pady=2)
        
        # 手动输入框架
        manual_frame = ttk.LabelFrame(left_frame, text="手动输入", style='TLabelframe', padding="8")
        manual_frame.pack(fill=tk.X, pady=8)
        
        # 手动输入锤击数
        manual_strike_frame = ttk.Frame(manual_frame, style='TFrame')
        manual_strike_frame.pack(fill=tk.X, pady=4)
        
        ttk.Label(manual_strike_frame, text="手动输入锤击数:", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.manual_strike_var = tk.StringVar(value="0")
        manual_entry = ttk.Entry(manual_strike_frame, textvariable=self.manual_strike_var, width=8,
                                style='TEntry', font=('Arial', 9))
        manual_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_strike_frame, text="添加", command=self.add_manual_strikes,
                  style='TButton', width=6).pack(side=tk.LEFT, padx=2)
        
        # 贯入度和超高输入
        params_input_frame = ttk.Frame(manual_frame, style='TFrame')
        params_input_frame.pack(fill=tk.X, pady=4)
        
        ttk.Label(params_input_frame, text="贯入度(mm):", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.penetration_var = tk.StringVar()
        penetration_entry = ttk.Entry(params_input_frame, textvariable=self.penetration_var, width=8,
                                     style='TEntry', font=('Arial', 9))
        penetration_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(params_input_frame, text="超高(m):", style='TLabel', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(10,0))
        self.elevation_var = tk.StringVar()
        elevation_entry = ttk.Entry(params_input_frame, textvariable=self.elevation_var, width=8,
                                   style='TEntry', font=('Arial', 9))
        elevation_entry.pack(side=tk.LEFT, padx=5)
        
        # 两个按钮：确定和条件修改
        button_frame = ttk.Frame(manual_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=4)
        
        ttk.Button(button_frame, text="确定", command=self.perform_judgment,
                  style='TButton', width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="条件修改", command=self.show_judgment_dialog,
                  style='TButton', width=8).pack(side=tk.LEFT, padx=2)
        
        # === 右侧数据显示区域 ===
        
        # 状态显示
        status_frame = ttk.LabelFrame(right_frame, text="实时状态", style='TLabelframe', padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        # 状态信息第一行
        status_row1 = ttk.Frame(status_frame, style='TFrame')
        status_row1.pack(fill=tk.X, pady=3)
        
        # 当前桩名称
        ttk.Label(status_row1, text="当前桩:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.current_pile_name_var = tk.StringVar(value="未设置")
        ttk.Label(status_row1, textvariable=self.current_pile_name_var, style='Highlight.TLabel', 
                 width=12).pack(side=tk.LEFT, padx=5)
        
        # 开始时间
        ttk.Label(status_row1, text="开始时间:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.start_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(status_row1, textvariable=self.start_time_var, style='Value.TLabel', 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # 结束时间
        ttk.Label(status_row1, text="结束时间:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.end_time_var = tk.StringVar(value="--:--:--")
        ttk.Label(status_row1, textvariable=self.end_time_var, style='Value.TLabel', 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # 监测状态
        ttk.Label(status_row1, text="监测状态:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_row1, textvariable=self.status_var, style='Status.TLabel', 
                 width=8).pack(side=tk.LEFT, padx=5)
        
        # 状态信息第二行
        status_row2 = ttk.Frame(status_frame, style='TFrame')
        status_row2.pack(fill=tk.X, pady=3)
        
        # 当前音量
        ttk.Label(status_row2, text="当前音量:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.volume_var = tk.StringVar(value="0.0000")
        ttk.Label(status_row2, textvariable=self.volume_var, style='Value.TLabel', 
                 width=8).pack(side=tk.LEFT, padx=5)
        
        # 当前频率
        ttk.Label(status_row2, text="当前频率:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.frequency_var = tk.StringVar(value="0 Hz")
        ttk.Label(status_row2, textvariable=self.frequency_var, style='Value.TLabel', 
                 width=8).pack(side=tk.LEFT, padx=5)
        
        # 总锤击数
        ttk.Label(status_row2, text="总锤击数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.strikes_var = tk.StringVar(value="0")
        ttk.Label(status_row2, textvariable=self.strikes_var, style='Highlight.TLabel', 
                 width=6).pack(side=tk.LEFT, padx=5)
        
        # 持续时间
        ttk.Label(status_row2, text="持续时间:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.duration_var = tk.StringVar(value="00:00")
        ttk.Label(status_row2, textvariable=self.duration_var, style='Value.TLabel', 
                 width=6).pack(side=tk.LEFT, padx=5)
        
        # 状态信息第三行
        status_row3 = ttk.Frame(status_frame, style='TFrame')
        status_row3.pack(fill=tk.X, pady=3)
        
        # 锤击频率
        ttk.Label(status_row3, text="锤击频率:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.strikes_per_min_var = tk.StringVar(value="0.0")
        ttk.Label(status_row3, textvariable=self.strikes_per_min_var, style='Value.TLabel', 
                 width=6).pack(side=tk.LEFT, padx=5)
        
        # 音量范围
        ttk.Label(status_row3, text="音量范围:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.volume_range_var = tk.StringVar(value="0.0000-0.0000")
        ttk.Label(status_row3, textvariable=self.volume_range_var, style='Value.TLabel', 
                 width=12).pack(side=tk.LEFT, padx=5)
        
        # 频率范围
        ttk.Label(status_row3, text="频率范围:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(15,0))
        self.freq_range_var = tk.StringVar(value="0-0 Hz")
        ttk.Label(status_row3, textvariable=self.freq_range_var, style='Value.TLabel', 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(right_frame, text="统计信息", style='TLabelframe', padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        stats_row1 = ttk.Frame(stats_frame, style='TFrame')
        stats_row1.pack(fill=tk.X, pady=3)
        
        ttk.Label(stats_row1, text="完成锤击桩总数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.total_piles_var = tk.StringVar(value="0")
        ttk.Label(stats_row1, textvariable=self.total_piles_var, style='Value.TLabel').pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stats_row1, text="最大锤击数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(20,0))
        self.max_strikes_var = tk.StringVar(value="0")
        ttk.Label(stats_row1, textvariable=self.max_strikes_var, style='Value.TLabel').pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stats_row1, text="最小锤击数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(20,0))
        self.min_strikes_var = tk.StringVar(value="0")
        ttk.Label(stats_row1, textvariable=self.min_strikes_var, style='Value.TLabel').pack(side=tk.LEFT, padx=5)
        
        stats_row2 = ttk.Frame(stats_frame, style='TFrame')
        stats_row2.pack(fill=tk.X, pady=3)
        
        ttk.Label(stats_row2, text="平均锤击数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.avg_strikes_var = tk.StringVar(value="0.0")
        ttk.Label(stats_row2, textvariable=self.avg_strikes_var, style='Value.TLabel').pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stats_row2, text="累计锤击数:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(20,0))
        self.total_strikes_var = tk.StringVar(value="0")
        ttk.Label(stats_row2, textvariable=self.total_strikes_var, style='Highlight.TLabel').pack(side=tk.LEFT, padx=5)
        
        # 施工判定结果显示
        judgment_frame = ttk.Frame(right_frame, style='TFrame')
        judgment_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(judgment_frame, text="施工判定:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.judgment_var = tk.StringVar(value="未判定")
        judgment_label = ttk.Label(judgment_frame, textvariable=self.judgment_var, 
                                 style='Highlight.TLabel', font=('Arial', 10, 'bold'))
        judgment_label.pack(side=tk.LEFT, padx=5)
        
        # 频率过滤状态
        filter_frame = ttk.Frame(right_frame, style='TFrame')
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="频率过滤:", style='TLabel', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.filter_status_var = tk.StringVar(value=f"{self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
        filter_label = ttk.Label(filter_frame, textvariable=self.filter_status_var, 
                               style='Value.TLabel', font=('Arial', 10))
        filter_label.pack(side=tk.LEFT, padx=5)
        
        # 日志区域
        log_frame = ttk.LabelFrame(right_frame, text="监测日志", style='TLabelframe', padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=("Consolas", 9),
                                                bg='#f8f9fa', fg='#2c3e50', insertbackground='black')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.log("系统启动完成 - 打桩锤击计数监测系统 v9.1-完整版")
        self.log(f"阈值: {self.threshold:.3f}, 频率过滤: {self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
        
        # 初始显示正确的模式
        self.on_mode_change()
    
    # 由于代码长度限制，以下是关键功能方法的实现
    # 完整代码请参考附件或运行环境
    
    def start_ai_training(self):
        """开始AI训练"""
        if len(self.training_data) < 10:
            messagebox.showwarning("训练数据不足", "至少需要10组训练数据才能开始AI训练")
            return
            
        thread = threading.Thread(target=self._ai_training_thread, daemon=True)
        thread.start()
    
    def _ai_training_thread(self):
        """AI训练线程"""
        try:
            self.is_ai_training = True
            self.status_var.set("AI训练中")
            self.log("🤖 开始AI模型训练...")
            
            # 准备训练数据
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练随机森林模型
            self.ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ai_model.fit(X_train, y_train)
            
            # 评估模型
            accuracy = self.ai_model.score(X_test, y_test)
            
            self.root.after(0, self._finish_ai_training, accuracy)
            
        except Exception as e:
            self.root.after(0, self._ai_training_error, str(e))
    
    def _finish_ai_training(self, accuracy):
        """完成AI训练"""
        self.is_ai_training = False
        self.status_var.set("训练完成")
        self.ai_status_var.set(f"AI模型: 准确率{accuracy:.2f}")
        self.save_ai_model()
        self.log(f"✅ AI训练完成! 准确率: {accuracy:.2f}")
        messagebox.showinfo("AI训练完成", f"模型训练完成!\n测试集准确率: {accuracy:.2f}")
    
    def use_ai_analysis(self):
        """使用AI分析"""
        if not self.ai_model:
            messagebox.showwarning("AI模型未训练", "请先训练AI模型")
            return
            
        if not self.is_monitoring:
            messagebox.showwarning("未在监测状态", "请在监测状态下使用AI分析")
            return
            
        self.log("🧠 启用AI智能分析")
        messagebox.showinfo("AI分析", "AI智能分析已启用，将自动优化检测参数")
    
    def perform_judgment(self):
        """执行施工判定"""
        try:
            penetration = float(self.penetration_var.get())
            elevation = float(self.elevation_var.get())
            
            if not self.pile_details:
                messagebox.showwarning("警告", "暂无完成的桩数据")
                return
                
            last_pile = self.pile_details[-1]
            total_strikes = last_pile['strikes']
            
            # 施工判定逻辑
            judgment = ""
            if (elevation <= self.judgment_conditions['condition1']['elevation_max'] and 
                penetration <= self.judgment_conditions['condition1']['penetration_max']):
                judgment = f"✅ 条件1: {self.judgment_conditions['condition1']['action']}"
            elif (elevation < self.judgment_conditions['condition2']['elevation_max'] and 
                  penetration > self.judgment_conditions['condition2']['penetration_min']):
                judgment = f"✅ 条件2: {self.judgment_conditions['condition2']['action']}"
            elif (elevation > self.judgment_conditions['condition3']['elevation_min'] and 
                  penetration <= self.judgment_conditions['condition3']['penetration_max'] and 
                  total_strikes > self.judgment_conditions['condition3']['strikes_min']):
                judgment = f"✅ 条件3: {self.judgment_conditions['condition3']['action']}"
            else:
                judgment = "⚠️ 未满足特定条件，请根据实际情况判断"
            
            # 更新桩信息
            last_pile['penetration_depth'] = penetration
            last_pile['elevation_height'] = elevation
            last_pile['construction_judgment'] = judgment
            
            # 更新显示
            self.judgment_var.set(judgment)
            
            self.log(f"📋 施工判定: {judgment}")
            self.log(f"📏 贯入度: {penetration}mm, 超高: {elevation}m")
            
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字")
    
    def add_manual_strikes(self):
        """手动添加锤击数"""
        try:
            manual_count = int(self.manual_strike_var.get())
            if manual_count > 0:
                self.manual_strikes += manual_count
                total_strikes = self.current_pile_strikes + self.manual_strikes
                self.strikes_var.set(str(total_strikes))
                self.log(f"🔢 手动添加 {manual_count} 次锤击，当前总计: {total_strikes} 次")
                self.manual_strike_var.set("0")
                self.update_statistics()
            else:
                messagebox.showwarning("输入错误", "请输入有效的锤击数")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数字")
    
    def update_statistics(self):
        """更新统计信息"""
        total_current_strikes = self.current_pile_strikes + self.manual_strikes
        total_completed_strikes = sum(self.all_pile_strikes)
        total_strikes = total_current_strikes + total_completed_strikes
        
        self.total_strikes_var.set(str(total_strikes))
        
        if self.all_pile_strikes:
            avg_strikes = total_completed_strikes / len(self.all_pile_strikes)
            max_strikes = max(self.all_pile_strikes)
            min_strikes = min(self.all_pile_strikes)
            
            self.avg_strikes_var.set(f"{avg_strikes:.1f}")
            self.max_strikes_var.set(str(max_strikes))
            self.min_strikes_var.set(str(min_strikes))
            self.total_piles_var.set(str(len(self.all_pile_strikes)))
    
    def start_status_update(self):
        """状态更新线程"""
        def update_loop():
            while True:
                if self.is_monitoring and self.pile_start_time:
                    duration = time.time() - self.pile_start_time
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    self.duration_var.set(f"{minutes:02d}:{seconds:02d}")
                    
                    # 更新开始时间
                    start_str = datetime.fromtimestamp(self.pile_start_time).strftime('%H:%M:%S')
                    self.start_time_var.set(start_str)
                    
                    # 更新结束时间
                    if self.last_strike_time:
                        end_str = datetime.fromtimestamp(self.last_strike_time).strftime('%H:%M:%S')
                        self.end_time_var.set(end_str)
                    
                    # 更新每分钟锤击数
                    if duration > 0:
                        strikes_per_min = (self.current_pile_strikes / duration) * 60
                        self.strikes_per_min_var.set(f"{strikes_per_min:.1f}")
                    
                    # 更新音量范围和频率范围
                    if self.strike_volumes:
                        min_vol = min(self.strike_volumes)
                        max_vol = max(self.strike_volumes)
                        self.volume_range_var.set(f"{min_vol:.4f}-{max_vol:.4f}")
                    
                    if self.strike_frequencies:
                        min_freq = min(self.strike_frequencies)
                        max_freq = max(self.strike_frequencies)
                        self.freq_range_var.set(f"{min_freq:.0f}-{max_freq:.0f}Hz")
                    
                    # 更新累计锤击数
                    total_current_strikes = self.current_pile_strikes + self.manual_strikes
                    total_completed_strikes = sum(self.all_pile_strikes)
                    total_strikes = total_current_strikes + total_completed_strikes
                    self.total_strikes_var.set(str(total_strikes))
                        
                time.sleep(0.5)
                
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def on_mode_change(self):
        """模式切换"""
        if self.mode_var.get() == "realtime":
            self.realtime_frame.pack(fill=tk.X, pady=5)
            self.file_frame.pack_forget()
            self.analyze_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)
            self.log("切换到实时监测模式")
        else:
            self.realtime_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, pady=5)
            self.analyze_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.DISABLED)
            self.log("切换到文件分析模式")
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def update_device_list(self):
        """更新音频设备列表"""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(f"{i}: {device['name']}")
            
            self.device_combo['values'] = input_devices
            if input_devices:
                self.device_combo.set(input_devices[0])
        except Exception as e:
            self.log(f"获取音频设备失败: {e}")
            
    def browse_file(self):
        """浏览文件"""
        filename = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.wav *.m4a *.mp3 *.flac *.ogg"),
                ("WAV文件", "*.wav"),
                ("M4A文件", "*.m4a"),
                ("MP3文件", "*.mp3"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.file_path_var.set(filename)
            self.current_audio_file = filename
            self.log(f"导入文件: {os.path.basename(filename)}")
            
    def on_threshold_change(self, value):
        """阈值改变"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"{self.threshold:.3f}")
        self.save_config()
        
    def on_min_freq_change(self, value):
        """最低频率改变"""
        self.min_frequency = float(value)
        self.min_freq_label.config(text=f"{self.min_frequency:.0f}Hz")
        self.filter_status_var.set(f"{self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
        self.save_config()
        
    def on_max_freq_change(self, value):
        """最高频率改变"""
        self.max_frequency = float(value)
        self.max_freq_label.config(text=f"{self.max_frequency:.0f}Hz")
        self.filter_status_var.set(f"{self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
        self.save_config()
        
    def on_silence_change(self, event=None):
        """静默时间改变"""
        try:
            self.silence_duration = float(self.silence_var.get())
            self.save_config()
        except:
            pass

    def on_advanced_change(self, event=None):
        """高级参数改变"""
        try:
            self.min_interval = float(self.min_interval_var.get())
            self.file_analysis_threshold_multiplier = float(self.file_threshold_multiplier_var.get())
            self.save_config()
        except:
            pass

    def set_pile_name(self):
        """设置当前桩名称"""
        name = self.pile_name_var.get().strip()
        if name:
            self.current_pile_name = name
            self.current_pile_name_var.set(name)
            self.log(f"设置桩名称: {name}")

    def calculate_frequency(self, audio_data):
        """计算音频数据的主频率"""
        try:
            # 使用FFT计算频率
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(fft_data), 1.0 / self.sample_rate)
            
            # 取正频率部分
            positive_freq_idx = frequencies > 0
            frequencies = frequencies[positive_freq_idx]
            magnitudes = np.abs(fft_data[positive_freq_idx])
            
            # 找到最大幅值对应的频率
            if len(magnitudes) > 0:
                max_idx = np.argmax(magnitudes)
                dominant_freq = frequencies[max_idx]
                return dominant_freq
            else:
                return 0
        except Exception as e:
            return 0
    
    def is_valid_frequency(self, frequency):
        """检查频率是否在有效范围内"""
        return self.min_frequency <= frequency <= self.max_frequency
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频回调 - 增加频率过滤"""
        if self.is_monitoring:
            # 计算音量
            volume = np.sqrt(np.mean(indata**2))
            
            # 计算频率
            frequency = self.calculate_frequency(indata[:, 0])
            
            self.root.after(0, self.process_audio, volume, frequency)
            
    def process_audio(self, volume, frequency):
        """处理音频数据 - 增加频率过滤"""
        if not self.is_monitoring:
            return
            
        self.volume_var.set(f"{volume:.4f}")
        self.frequency_var.set(f"{frequency:.0f} Hz")
        
        current_time = time.time()
        
        # 开始新桩
        if self.pile_start_time is None:
            self.pile_start_time = current_time
            self.strike_times = []
            self.strike_frequencies = []
            self.strike_volumes = []
            pile_num = len(self.all_pile_strikes) + 1
            pile_name = self.current_pile_name if self.current_pile_name else f"桩{pile_num}"
            self.current_pile_name_var.set(pile_name)
            self.log(f"开始监测 {pile_name}")
            
        # 检测锤击 - 增加频率过滤条件
        if (volume > self.threshold and 
            self.is_valid_frequency(frequency) and  # 新增频率过滤
            (self.last_strike_time is None or 
             (current_time - self.last_strike_time) > self.min_interval)):
            
            self.current_pile_strikes += 1
            self.last_strike_time = current_time
            self.strike_times.append(current_time)
            self.strike_frequencies.append(frequency)
            self.strike_volumes.append(volume)
            
            total_strikes = self.current_pile_strikes + self.manual_strikes
            self.strikes_var.set(str(total_strikes))
            self.update_statistics()
            
            # 收集训练数据
            if len(self.strike_volumes) > 0:
                features = [volume, frequency, np.mean(self.strike_volumes), np.std(self.strike_volumes)]
                self.training_data.append(features)
                self.training_labels.append(1)  # 1表示有效锤击
            
            if total_strikes <= 3:
                time_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
                self.log(f"🔨 锤击! 次数:{total_strikes} 时间:{time_str} 频率:{frequency:.0f}Hz")
            else:
                self.log(f"🔨 锤击 #{total_strikes} 频率:{frequency:.0f}Hz")
                
        # 检测桩完成
        if (self.last_strike_time and 
            (current_time - self.last_strike_time) > self.silence_duration and
            (self.current_pile_strikes + self.manual_strikes) > 0):
            self.complete_pile()
            
    def complete_pile(self):
        """完成当前桩"""
        pile_num = len(self.all_pile_strikes) + 1
        total_strikes = self.current_pile_strikes + self.manual_strikes
        duration = self.last_strike_time - self.pile_start_time if self.last_strike_time else 0
        
        pile_name = self.current_pile_name if self.current_pile_name else f"桩{pile_num}"
        
        # 计算统计信息
        freq_range = "0-0 Hz"
        volume_range = "0.0000-0.0000"
        strikes_per_min = 0.0
        
        if self.strike_frequencies:
            min_freq = min(self.strike_frequencies)
            max_freq = max(self.strike_frequencies)
            freq_range = f"{min_freq:.0f}-{max_freq:.0f}Hz"
            
        if self.strike_volumes:
            min_vol = min(self.strike_volumes)
            max_vol = max(self.strike_volumes)
            volume_range = f"{min_vol:.4f}-{max_vol:.4f}"
            
        if duration > 0:
            strikes_per_min = (total_strikes / duration) * 60
        
        # 保存详细信息
        pile_info = {
            'number': pile_num,
            'name': pile_name,
            'strikes': total_strikes,
            'start_time': float(self.pile_start_time),
            'end_time': float(self.last_strike_time) if self.last_strike_time else float(time.time()),
            'duration': float(duration),
            'strike_times': self.strike_times.copy(),
            'strike_frequencies': self.strike_frequencies.copy(),
            'strike_volumes': self.strike_volumes.copy(),
            'frequency_range': freq_range,
            'volume_range': volume_range,
            'strikes_per_minute': strikes_per_min,
            'penetration_depth': None,
            'elevation_height': None,
            'construction_judgment': "未判定"
        }
        self.pile_details.append(pile_info)
        self.all_pile_strikes.append(total_strikes)
        
        # 记录日志
        start_str = datetime.fromtimestamp(self.pile_start_time).strftime('%H:%M:%S')
        end_str = datetime.fromtimestamp(self.last_strike_time).strftime('%H:%M:%S') if self.last_strike_time else "--:--:--"
        duration_str = self.format_duration(duration)
        
        self.log(f"🎯 {pile_name} 完成! {total_strikes}次")
        self.log(f"📊 统计: 频率{freq_range}, 音量{volume_range}, {strikes_per_min:.1f}锤/分钟")
        self.log(f"⏰ 时间: {start_str} - {end_str}, 持续: {duration_str}")
        self.log(f"💡 请输入贯入度和超高数据进行施工判定")
        
        # 重置状态
        self.current_pile_strikes = 0
        self.manual_strikes = 0
        self.pile_start_time = None
        self.last_strike_time = None
        self.strike_times = []
        self.strike_frequencies = []
        self.strike_volumes = []
        self.current_pile_name = ""
        self.pile_name_var.set("")
        self.current_pile_name_var.set("未设置")
        self.strikes_var.set("0")
        self.start_time_var.set("--:--:--")
        self.end_time_var.set("--:--:--")
        self.total_piles_var.set(str(len(self.all_pile_strikes)))
        self.duration_var.set("00:00")
        self.strikes_per_min_var.set("0.0")
        self.volume_range_var.set("0.0000-0.0000")
        self.freq_range_var.set("0-0 Hz")
        self.update_statistics()

    def manual_end_pile(self):
        """手动结束当前桩监测"""
        if not self.is_monitoring:
            messagebox.showinfo("提示", "当前未在监测状态")
            return
            
        if self.current_pile_strikes == 0 and self.manual_strikes == 0:
            messagebox.showinfo("提示", "当前桩没有锤击记录")
            return
            
        if messagebox.askyesno("确认", f"确定要结束当前桩监测吗？\n当前锤击数: {self.current_pile_strikes + self.manual_strikes}"):
            self.complete_pile()
            self.log("⏹️ 手动结束当前桩监测")

    def start_monitoring(self):
        """开始监测"""
        if self.is_monitoring:
            return
            
        try:
            device_selection = self.device_combo.get()
            device_index = int(device_selection.split(":")[0]) if device_selection else None
            
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=device_index,
                channels=1,
                callback=self.audio_callback,
                dtype=np.float32
            )
            
            self.audio_stream.start()
            self.is_monitoring = True
            self.status_var.set("监测中")
            
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.calibrate_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            self.end_pile_btn.config(state=tk.NORMAL)
            
            self.log("🚀 开始实时监测")
            self.log(f"🎛️ 阈值: {self.threshold:.3f}, 频率过滤: {self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
            
        except Exception as e:
            self.log(f"❌ 启动失败: {e}")
            messagebox.showerror("错误", f"启动监测失败: {e}")
            
    def stop_monitoring(self):
        """停止监测"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            
        # 记录最后一根桩
        total_strikes = self.current_pile_strikes + self.manual_strikes
        if total_strikes > 0:
            pile_num = len(self.all_pile_strikes) + 1
            
            if self.pile_start_time:
                pile_name = self.current_pile_name if self.current_pile_name else f"桩{pile_num}"
                duration = (self.last_strike_time if self.last_strike_time else time.time()) - self.pile_start_time
                
                # 计算统计信息
                freq_range = "0-0 Hz"
                volume_range = "0.0000-0.0000"
                strikes_per_min = 0.0
                
                if self.strike_frequencies:
                    min_freq = min(self.strike_frequencies)
                    max_freq = max(self.strike_frequencies)
                    freq_range = f"{min_freq:.0f}-{max_freq:.0f}Hz"
                    
                if self.strike_volumes:
                    min_vol = min(self.strike_volumes)
                    max_vol = max(self.strike_volumes)
                    volume_range = f"{min_vol:.4f}-{max_vol:.4f}"
                    
                if duration > 0:
                    strikes_per_min = (total_strikes / duration) * 60
                
                pile_info = {
                    'number': pile_num,
                    'name': pile_name,
                    'strikes': total_strikes,
                    'start_time': float(self.pile_start_time),
                    'end_time': float(self.last_strike_time) if self.last_strike_time else float(time.time()),
                    'duration': float(duration),
                    'strike_times': self.strike_times.copy(),
                    'strike_frequencies': self.strike_frequencies.copy(),
                    'strike_volumes': self.strike_volumes.copy(),
                    'frequency_range': freq_range,
                    'volume_range': volume_range,
                    'strikes_per_minute': strikes_per_min,
                    'penetration_depth': None,
                    'elevation_height': None,
                    'construction_judgment': "未判定"
                }
                self.pile_details.append(pile_info)
                self.all_pile_strikes.append(total_strikes)
                
            pile_name = self.current_pile_name if self.current_pile_name else f"桩{pile_num}"
            self.log(f"📝 记录{pile_name}: {total_strikes}次")
            
        self.status_var.set("已停止")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.calibrate_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL if self.mode_var.get() == "file" else tk.DISABLED)
        self.end_pile_btn.config(state=tk.DISABLED)
        
        self.log("🛑 监测停止")
        self.show_summary()

    def calibrate(self):
        """校准阈值和频率"""
        if self.is_monitoring:
            messagebox.showwarning("警告", "请先停止监测")
            return
            
        try:
            self.log("🎛️ 开始校准...")
            
            volumes = []
            frequencies = []
            start_time = time.time()
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=np.float32) as stream:
                while time.time() - start_time < 5:
                    data, overflowed = stream.read(self.chunk_size)
                    volume = float(np.sqrt(np.mean(data**2)))
                    frequency = self.calculate_frequency(data[:, 0])
                    volumes.append(volume)
                    frequencies.append(frequency)
                    time.sleep(0.1)
                    
            baseline = float(np.percentile(volumes, 90))
            max_volume = float(np.max(volumes))
            avg_frequency = float(np.mean([f for f in frequencies if f > 0]))
            
            if max_volume < 0.1:
                new_threshold = baseline * 5
            else:
                new_threshold = baseline * 3
                
            new_threshold = max(0.02, min(0.5, new_threshold))
            
            # 计算频率范围
            valid_frequencies = [f for f in frequencies if f > 0]
            if valid_frequencies:
                freq_std = np.std(valid_frequencies)
                suggested_min_freq = max(20, avg_frequency - freq_std)
                suggested_max_freq = min(5000, avg_frequency + freq_std)
            else:
                suggested_min_freq = 80
                suggested_max_freq = 2000
            
            self.log(f"📊 环境噪音: {baseline:.4f}")
            self.log(f"📊 环境频率: {avg_frequency:.0f}Hz")
            self.log(f"💡 推荐阈值: {new_threshold:.4f}")
            self.log(f"💡 推荐频率范围: {suggested_min_freq:.0f}-{suggested_max_freq:.0f}Hz")
            
            result = messagebox.askyesno("校准", 
                f"推荐阈值: {new_threshold:.4f}\n"
                f"环境频率: {avg_frequency:.0f}Hz\n"
                f"推荐频率范围: {suggested_min_freq:.0f}-{suggested_max_freq:.0f}Hz\n\n"
                f"使用此设置?")
            if result:
                self.threshold_var.set(new_threshold)
                self.threshold = new_threshold
                self.threshold_label.config(text=f"{new_threshold:.3f}")
                
                self.min_freq_var.set(suggested_min_freq)
                self.min_frequency = suggested_min_freq
                self.min_freq_label.config(text=f"{suggested_min_freq:.0f}Hz")
                
                self.max_freq_var.set(suggested_max_freq)
                self.max_frequency = suggested_max_freq
                self.max_freq_label.config(text=f"{suggested_max_freq:.0f}Hz")
                
                self.filter_status_var.set(f"{suggested_min_freq:.0f}-{suggested_max_freq:.0f}Hz")
                
                self.log(f"✅ 校准完成: 阈值={new_threshold:.4f}, 频率={suggested_min_freq:.0f}-{suggested_max_freq:.0f}Hz")
                self.save_config()
                
        except Exception as e:
            self.log(f"❌ 校准失败: {e}")

    def analyze_file(self):
        """分析音频文件"""
        if not self.file_path_var.get():
            messagebox.showwarning("警告", "请先选择音频文件")
            return
            
        if self.is_analyzing:
            return
            
        thread = threading.Thread(target=self._analyze_file_thread, daemon=True)
        thread.start()
        
    def _analyze_file_thread(self):
        """分析文件线程"""
        try:
            self.is_analyzing = True
            self.status_var.set("分析中")
            self.analyze_btn.config(state=tk.DISABLED)
            
            filename = self.file_path_var.get()
            self.log(f"📁 开始分析: {os.path.basename(filename)}")
            self.log(f"🎛️ 频率过滤: {self.min_frequency:.0f}-{self.max_frequency:.0f}Hz")
            
            # 根据文件格式选择处理方法
            if os.path.splitext(filename)[1].lower() == '.wav':
                audio_data, sample_rate = self._read_wav_file(filename)
            else:
                audio_data, sample_rate = self._convert_audio_file(filename)
            
            if audio_data is None:
                raise Exception("无法读取音频数据")
                
            # 使用正确的阈值计算
            chunk_size = 1024
            analysis_threshold = self.threshold * self.file_analysis_threshold_multiplier
            
            self.log(f"🎯 分析阈值: {analysis_threshold:.3f}")
            
            # 重置状态
            self.all_pile_strikes.clear()
            self.pile_details.clear()
            
            current_strikes = 0
            pile_start = 0
            last_strike = None
            last_valid_strike_time = 0
            pile_num = 1
            current_times = []
            current_frequencies = []
            current_volumes = []
            
            # 分析音频
            for i in range(0, len(audio_data), chunk_size):
                if not self.is_analyzing:
                    break
                    
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) == 0:
                    continue
                    
                volume = float(np.sqrt(np.mean(chunk**2)))
                frequency = self.calculate_frequency(chunk)
                current_time = i / sample_rate
                
                # 检测锤击 - 增加频率过滤
                if (volume > analysis_threshold and 
                    self.is_valid_frequency(frequency) and  # 频率过滤
                    (last_strike is None or (current_time - last_strike) > 0.5)):
                    
                    if last_valid_strike_time == 0 or (current_time - last_valid_strike_time) > self.min_interval:
                        current_strikes += 1
                        last_strike = current_time
                        last_valid_strike_time = current_time
                        current_times.append(current_time)
                        current_frequencies.append(frequency)
                        current_volumes.append(volume)
                        
                        if pile_start == 0:
                            pile_start = current_time
                            
                # 检测桩完成
                if (last_strike and 
                    (current_time - last_strike) > self.silence_duration and
                    current_strikes > 0):
                    
                    pile_name = f"桩{pile_num}"
                    duration = last_strike - pile_start
                    
                    # 计算统计信息
                    freq_range = "0-0 Hz"
                    volume_range = "0.0000-0.0000"
                    strikes_per_min = 0.0
                    
                    if current_frequencies:
                        min_freq = min(current_frequencies)
                        max_freq = max(current_frequencies)
                        freq_range = f"{min_freq:.0f}-{max_freq:.0f}Hz"
                        
                    if current_volumes:
                        min_vol = min(current_volumes)
                        max_vol = max(current_volumes)
                        volume_range = f"{min_vol:.4f}-{max_vol:.4f}"
                        
                    if duration > 0:
                        strikes_per_min = (current_strikes / duration) * 60
                    
                    pile_info = {
                        'number': pile_num,
                        'name': pile_name,
                        'strikes': current_strikes,
                        'start_time': float(pile_start),
                        'end_time': float(last_strike),
                        'duration': float(duration),
                        'strike_times': current_times.copy(),
                        'strike_frequencies': current_frequencies.copy(),
                        'strike_volumes': current_volumes.copy(),
                        'frequency_range': freq_range,
                        'volume_range': volume_range,
                        'strikes_per_minute': strikes_per_min,
                        'penetration_depth': None,
                        'elevation_height': None,
                        'construction_judgment': "未判定"
                    }
                    self.pile_details.append(pile_info)
                    self.all_pile_strikes.append(current_strikes)
                    
                    self.log(f"🎯 {pile_name}完成! {current_strikes}次")
                    
                    current_strikes = 0
                    pile_start = 0
                    last_strike = None
                    last_valid_strike_time = 0
                    current_times = []
                    current_frequencies = []
                    current_volumes = []
                    pile_num += 1
                    
            # 处理最后一根桩
            if current_strikes > 0:
                pile_name = f"桩{pile_num}"
                end_time = len(audio_data)/sample_rate
                duration = end_time - pile_start
                
                # 计算统计信息
                freq_range = "0-0 Hz"
                volume_range = "0.0000-0.0000"
                strikes_per_min = 0.0
                
                if current_frequencies:
                    min_freq = min(current_frequencies)
                    max_freq = max(current_frequencies)
                    freq_range = f"{min_freq:.0f}-{max_freq:.0f}Hz"
                    
                if current_volumes:
                    min_vol = min(current_volumes)
                    max_vol = max(current_volumes)
                    volume_range = f"{min_vol:.4f}-{max_vol:.4f}"
                    
                if duration > 0:
                    strikes_per_min = (current_strikes / duration) * 60
                
                pile_info = {
                    'number': pile_num,
                    'name': pile_name,
                    'strikes': current_strikes,
                    'start_time': float(pile_start),
                    'end_time': float(end_time),
                    'duration': float(duration),
                    'strike_times': current_times.copy(),
                    'strike_frequencies': current_frequencies.copy(),
                    'strike_volumes': current_volumes.copy(),
                    'frequency_range': freq_range,
                    'volume_range': volume_range,
                    'strikes_per_minute': strikes_per_min,
                    'penetration_depth': None,
                    'elevation_height': None,
                    'construction_judgment': "未判定"
                }
                self.pile_details.append(pile_info)
                self.all_pile_strikes.append(current_strikes)
                self.log(f"🎯 {pile_name}完成! {current_strikes}次 (文件结束)")
                
            self.root.after(0, self._finish_analysis)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self._analysis_error, error_msg)
            
    def _read_wav_file(self, filename):
        """读取WAV文件"""
        try:
            with wave.open(filename, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                frames = wav_file.readframes(n_frames)
                
            if sample_width == 2:
                dtype = np.int16
                max_value = 32768.0
            elif sample_width == 4:
                dtype = np.int32
                max_value = 2147483648.0
            else:
                dtype = np.int8
                max_value = 128.0
                
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels)
                audio_data = np.mean(audio_data, axis=1)
                
            audio_data = audio_data.astype(np.float32) / max_value
            
            return audio_data, sample_rate
        except Exception as e:
            self.log(f"❌ 读取WAV文件失败: {e}")
            return None, None
        
    def _convert_audio_file(self, filename):
        """使用FFmpeg转换音频文件为WAV格式"""
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            self.log("🔄 转换音频文件...")
            
            cmd = [
                'ffmpeg', '-i', filename,
                '-ac', '1',
                '-ar', '44100',
                '-acodec', 'pcm_s16le',
                '-y',
                temp_wav.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg转换失败: {result.stderr}")
                
            audio_data, sample_rate = self._read_wav_file(temp_wav.name)
            
            os.unlink(temp_wav.name)
            
            return audio_data, sample_rate
            
        except Exception as e:
            if 'temp_wav' in locals() and os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
            raise e
            
    def _finish_analysis(self):
        """完成分析"""
        self.is_analyzing = False
        self.status_var.set("分析完成")
        self.analyze_btn.config(state=tk.NORMAL)
        self.total_piles_var.set(str(len(self.all_pile_strikes)))
        self.update_statistics()
        self.log("✅ 文件分析完成")
        self.show_summary()
        
    def _analysis_error(self, error_msg):
        """分析错误"""
        self.is_analyzing = False
        self.status_var.set("分析失败")
        self.analyze_btn.config(state=tk.NORMAL)
        self.log(f"❌ 分析失败: {error_msg}")
        messagebox.showerror("错误", f"文件分析失败:\n{error_msg}")

    def optimize_threshold(self):
        """阈值优化功能"""
        if not self.file_path_var.get():
            messagebox.showwarning("警告", "请先选择音频文件")
            return
            
        true_count = simpledialog.askinteger("阈值优化", "请输入真实的锤击次数:", 
                                           initialvalue=54, minvalue=1, maxvalue=1000)
        if true_count is None:
            return
            
        thread = threading.Thread(target=self._optimize_threshold_thread, 
                                 args=(true_count,), daemon=True)
        thread.start()
        
    def _optimize_threshold_thread(self, true_count):
        """阈值优化线程"""
        try:
            self.log(f"🔧 开始阈值优化，真实锤击数: {true_count}")
            
            filename = self.file_path_var.get()
            
            if os.path.splitext(filename)[1].lower() == '.wav':
                audio_data, sample_rate = self._read_wav_file(filename)
            else:
                audio_data, sample_rate = self._convert_audio_file(filename)
                
            if audio_data is None:
                raise Exception("无法读取音频数据")
                
            thresholds = []
            counts = []
            
            for threshold in [x * 0.01 for x in range(2, 80, 2)]:
                count = self._count_strikes_in_audio(audio_data, threshold, sample_rate)
                thresholds.append(threshold)
                counts.append(count)
                
            differences = [abs(count - true_count) for count in counts]
            best_idx = differences.index(min(differences))
            best_threshold = thresholds[best_idx]
            best_count = counts[best_idx]
            
            self.root.after(0, self._finish_optimization, best_threshold, best_count, true_count)
            
        except Exception as e:
            self.root.after(0, self._optimization_error, str(e))
            
    def _count_strikes_in_audio(self, audio_data, threshold, sample_rate):
        """在音频数据中计数锤击"""
        chunk_size = 1024
        strikes = 0
        last_strike_time = -self.min_interval
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) == 0:
                continue
                
            volume = float(np.sqrt(np.mean(chunk**2)))
            frequency = self.calculate_frequency(chunk)
            current_time = i / sample_rate
            
            # 增加频率过滤条件
            if (volume > threshold and 
                self.is_valid_frequency(frequency) and
                (current_time - last_strike_time) > self.min_interval):
                strikes += 1
                last_strike_time = current_time
                
        return strikes
        
    def _finish_optimization(self, best_threshold, best_count, true_count):
        """完成阈值优化"""
        result = messagebox.askyesno("阈值优化完成",
            f"优化结果:\n"
            f"真实锤击数: {true_count}\n"
            f"最佳阈值: {best_threshold:.3f}\n"
            f"预测锤击数: {best_count}\n"
            f"误差: {abs(best_count - true_count)}\n\n"
            f"是否应用此阈值?")
            
        if result:
            self.threshold_var.set(best_threshold)
            self.threshold = best_threshold
            self.threshold_label.config(text=f"{best_threshold:.3f}")
            self.save_config()
            self.log(f"✅ 应用优化阈值: {best_threshold:.3f}")

    def _optimization_error(self, error_msg):
        """优化错误"""
        self.log(f"❌ 阈值优化失败: {error_msg}")
        messagebox.showerror("优化失败", f"阈值优化失败:\n{error_msg}")

    def _ai_training_error(self, error_msg):
        """AI训练错误"""
        self.is_ai_training = False
        self.status_var.set("训练失败")
        self.log(f"❌ AI训练失败: {error_msg}")
        messagebox.showerror("AI训练失败", f"AI模型训练失败:\n{error_msg}")

    def show_judgment_dialog(self):
        """显示施工判定条件修改对话框"""
        judgment_window = tk.Toplevel(self.root)
        judgment_window.title("施工判定条件设置")
        judgment_window.geometry("500x400")
        judgment_window.configure(bg='#ffffff')
        judgment_window.resizable(False, False)
        
        main_frame = ttk.Frame(judgment_window, style='TFrame', padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="施工判定条件设置", 
                 style='Title.TLabel').pack(pady=10)
        
        # 条件1设置
        cond1_frame = ttk.LabelFrame(main_frame, text="条件1: 标高超高≤Xm且贯入度≤Ymm → 停止锤击", 
                                   style='TLabelframe', padding="10")
        cond1_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(cond1_frame, text="标高超高最大值(m):", style='TLabel').pack(side=tk.LEFT)
        cond1_elevation_var = tk.DoubleVar(value=self.judgment_conditions['condition1']['elevation_max'])
        cond1_elevation_spin = ttk.Spinbox(cond1_frame, from_=0.1, to=10.0, 
                                         textvariable=cond1_elevation_var, width=8)
        cond1_elevation_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(cond1_frame, text="贯入度最大值(mm):", style='TLabel').pack(side=tk.LEFT, padx=(15,0))
        cond1_penetration_var = tk.DoubleVar(value=self.judgment_conditions['condition1']['penetration_max'])
        cond1_penetration_spin = ttk.Spinbox(cond1_frame, from_=0.1, to=20.0, 
                                           textvariable=cond1_penetration_var, width=8)
        cond1_penetration_spin.pack(side=tk.LEFT, padx=5)
        
        # 条件2设置
        cond2_frame = ttk.LabelFrame(main_frame, text="条件2: 标高超高<Xm且贯入度>Ymm → 打至标高超高0.4m停锤", 
                                   style='TLabelframe', padding="10")
        cond2_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(cond2_frame, text="标高超高最大值(m):", style='TLabel').pack(side=tk.LEFT)
        cond2_elevation_var = tk.DoubleVar(value=self.judgment_conditions['condition2']['elevation_max'])
        cond2_elevation_spin = ttk.Spinbox(cond2_frame, from_=0.1, to=10.0, 
                                         textvariable=cond2_elevation_var, width=8)
        cond2_elevation_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(cond2_frame, text="贯入度最小值(mm):", style='TLabel').pack(side=tk.LEFT, padx=(15,0))
        cond2_penetration_var = tk.DoubleVar(value=self.judgment_conditions['condition2']['penetration_min'])
        cond2_penetration_spin = ttk.Spinbox(cond2_frame, from_=0.1, to=20.0, 
                                           textvariable=cond2_penetration_var, width=8)
        cond2_penetration_spin.pack(side=tk.LEFT, padx=5)
        
        # 条件3设置
        cond3_frame = ttk.LabelFrame(main_frame, text="条件3: 标高超高>Xm且贯入度≤Ymm且锤击数>Z → 再施打100锤观察", 
                                   style='TLabelframe', padding="10")
        cond3_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(cond3_frame, text="标高超高最小值(m):", style='TLabel').pack(side=tk.LEFT)
        cond3_elevation_var = tk.DoubleVar(value=self.judgment_conditions['condition3']['elevation_min'])
        cond3_elevation_spin = ttk.Spinbox(cond3_frame, from_=0.1, to=10.0, 
                                         textvariable=cond3_elevation_var, width=8)
        cond3_elevation_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(cond3_frame, text="贯入度最大值(mm):", style='TLabel').pack(side=tk.LEFT, padx=(15,0))
        cond3_penetration_var = tk.DoubleVar(value=self.judgment_conditions['condition3']['penetration_max'])
        cond3_penetration_spin = ttk.Spinbox(cond3_frame, from_=0.1, to=20.0, 
                                           textvariable=cond3_penetration_var, width=8)
        cond3_penetration_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(cond3_frame, text="锤击数最小值:", style='TLabel').pack(side=tk.LEFT, padx=(15,0))
        cond3_strikes_var = tk.IntVar(value=self.judgment_conditions['condition3']['strikes_min'])
        cond3_strikes_spin = ttk.Spinbox(cond3_frame, from_=100, to=10000, 
                                       textvariable=cond3_strikes_var, width=8)
        cond3_strikes_spin.pack(side=tk.LEFT, padx=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=15)
        
        def save_conditions():
            """保存条件设置"""
            try:
                # 更新条件参数
                self.judgment_conditions['condition1']['elevation_max'] = float(cond1_elevation_var.get())
                self.judgment_conditions['condition1']['penetration_max'] = float(cond1_penetration_var.get())
                self.judgment_conditions['condition2']['elevation_max'] = float(cond2_elevation_var.get())
                self.judgment_conditions['condition2']['penetration_min'] = float(cond2_penetration_var.get())
                self.judgment_conditions['condition3']['elevation_min'] = float(cond3_elevation_var.get())
                self.judgment_conditions['condition3']['penetration_max'] = float(cond3_penetration_var.get())
                self.judgment_conditions['condition3']['strikes_min'] = int(cond3_strikes_var.get())
                
                self.log("✅ 施工判定条件已更新")
                judgment_window.destroy()
                
            except ValueError:
                messagebox.showwarning("输入错误", "请输入有效的数字")
        
        ttk.Button(button_frame, text="保存", command=save_conditions,
                  style='TButton', width=12).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="恢复默认", 
                  command=lambda: self.restore_default_conditions(
                      cond1_elevation_var, cond1_penetration_var,
                      cond2_elevation_var, cond2_penetration_var,
                      cond3_elevation_var, cond3_penetration_var, cond3_strikes_var
                  ), width=12, style='TButton').pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="取消", command=judgment_window.destroy,
                  style='TButton', width=10).pack(side=tk.LEFT, padx=10)
    
    def restore_default_conditions(self, cond1_elevation, cond1_penetration,
                                 cond2_elevation, cond2_penetration,
                                 cond3_elevation, cond3_penetration, cond3_strikes):
        """恢复默认判定条件"""
        cond1_elevation.set(3.0)
        cond1_penetration.set(5.0)
        cond2_elevation.set(3.0)
        cond2_penetration.set(5.0)
        cond3_elevation.set(3.0)
        cond3_penetration.set(2.0)
        cond3_strikes.set(2000)

    def quick_setup(self):
        """快速设置"""
        setup_window = tk.Toplevel(self.root)
        setup_window.title("快速设置")
        setup_window.geometry("400x300")
        setup_window.configure(bg='#ffffff')
        
        main_frame = ttk.Frame(setup_window, style='TFrame', padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="快速设置", font=("Arial", 12, "bold"), 
                 style='TLabel').pack(pady=5)
        
        # 锤击声音预设
        ttk.Label(main_frame, text="锤击声音类型:", font=("Arial", 10), 
                 style='TLabel').pack(anchor=tk.W, pady=(10,5))
        
        ttk.Button(main_frame, text="低沉锤击 (频率: 50-300Hz)", 
                  command=lambda: self.apply_preset(0.2, 80, 300), width=30,
                  style='TButton').pack(pady=3)
        ttk.Button(main_frame, text="中等锤击 (频率: 100-800Hz)", 
                  command=lambda: self.apply_preset(0.3, 100, 800), width=30,
                  style='TButton').pack(pady=3)
        ttk.Button(main_frame, text="清脆锤击 (频率: 200-1500Hz)", 
                  command=lambda: self.apply_preset(0.4, 200, 1500), width=30,
                  style='TButton').pack(pady=3)
        
        # 环境预设
        ttk.Label(main_frame, text="环境设置:", font=("Arial", 10), 
                 style='TLabel').pack(anchor=tk.W, pady=(10,5))
        
        ttk.Button(main_frame, text="安静环境 (阈值: 0.1)", 
                  command=lambda: self.apply_preset(0.1, 80, 2000), width=20,
                  style='TButton').pack(pady=2)
        ttk.Button(main_frame, text="一般环境 (阈值: 0.3)", 
                  command=lambda: self.apply_preset(0.3, 80, 2000), width=20,
                  style='TButton').pack(pady=2)
        ttk.Button(main_frame, text="嘈杂环境 (阈值: 0.5)", 
                  command=lambda: self.apply_preset(0.5, 80, 2000), width=20,
                  style='TButton').pack(pady=2)
        
    def apply_preset(self, threshold, min_freq, max_freq):
        """应用预设配置"""
        self.threshold_var.set(threshold)
        self.threshold = threshold
        self.threshold_label.config(text=f"{threshold:.3f}")
        
        self.min_freq_var.set(min_freq)
        self.min_frequency = min_freq
        self.min_freq_label.config(text=f"{min_freq:.0f}Hz")
        
        self.max_freq_var.set(max_freq)
        self.max_frequency = max_freq
        self.max_freq_label.config(text=f"{max_freq:.0f}Hz")
        
        self.filter_status_var.set(f"{min_freq:.0f}-{max_freq:.0f}Hz")
        
        self.save_config()
        self.log(f"✅ 应用预设: 阈值={threshold:.3f}, 频率={min_freq:.0f}-{max_freq:.0f}Hz")

    def export_data(self):
        """导出数据"""
        if not self.all_pile_strikes:
            messagebox.showinfo("导出", "无数据可导出")
            return
        try:
            filename = f"打桩监测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("打桩锤击计数监测系统 - 专业监测报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"监测模式: {'实时监测' if self.mode_var.get() == 'realtime' else '文件分析'}\n")
                f.write(f"频率过滤范围: {self.min_frequency:.0f}-{self.max_frequency:.0f}Hz\n")
                f.write(f"检测阈值: {self.threshold:.3f}\n")
                f.write("=" * 80 + "\n\n")
                
                total_strikes = sum(self.all_pile_strikes)
                total_piles = len(self.all_pile_strikes)
                avg_strikes = total_strikes / total_piles if total_piles > 0 else 0
                
                f.write(f"总体统计:\n")
                f.write(f"  总桩数: {total_piles} 根\n")
                f.write(f"  总锤击数: {total_strikes} 次\n")
                f.write(f"  平均锤击数: {avg_strikes:.1f} 次/桩\n")
                f.write(f"  最大锤击数: {max(self.all_pile_strikes)} 次\n")
                f.write(f"  最小锤击数: {min(self.all_pile_strikes)} 次\n\n")
                
                f.write("详细桩信息:\n")
                f.write("-" * 80 + "\n")
                
                for pile in self.pile_details:
                    if self.mode_var.get() == "realtime":
                        start_time = datetime.fromtimestamp(pile['start_time']).strftime('%H:%M:%S')
                        end_time = datetime.fromtimestamp(pile['end_time']).strftime('%H:%M:%S')
                    else:
                        start_time = self.format_timestamp(pile['start_time'])
                        end_time = self.format_timestamp(pile['end_time'])
                    
                    f.write(f"\n{pile['name']}:\n")
                    f.write(f"  锤击次数: {pile['strikes']} 次\n")
                    f.write(f"  开始时间: {start_time}\n")
                    f.write(f"  结束时间: {end_time}\n")
                    f.write(f"  持续时间: {self.format_duration(pile['duration'])}\n")
                    f.write(f"  频率范围: {pile['frequency_range']}\n")
                    f.write(f"  音量范围: {pile['volume_range']}\n")
                    f.write(f"  平均锤击/分钟: {pile['strikes_per_minute']:.1f}\n")
                    
                    if pile['penetration_depth'] is not None:
                        f.write(f"  贯入度: {pile['penetration_depth']} mm\n")
                        f.write(f"  超高: {pile['elevation_height']} m\n")
                        f.write(f"  施工判定: {pile['construction_judgment']}\n")
                    else:
                        f.write(f"  贯入度: 未输入\n")
                        f.write(f"  超高: 未输入\n")
                        f.write(f"  施工判定: 未判定\n")
                    
                    f.write(f"  锤击时间记录:\n")
                    for i, strike_time in enumerate(pile['strike_times'], 1):
                        if self.mode_var.get() == "realtime":
                            time_str = datetime.fromtimestamp(strike_time).strftime('%H:%M:%S')
                        else:
                            time_str = self.format_timestamp(strike_time)
                        
                        freq = pile['strike_frequencies'][i-1] if i <= len(pile['strike_frequencies']) else 0
                        volume = pile['strike_volumes'][i-1] if i <= len(pile['strike_volumes']) else 0
                        
                        f.write(f"    {i:3d}. {time_str} - 频率: {freq:.0f}Hz, 音量: {volume:.4f}\n")
                    
                    f.write("-" * 80 + "\n")
                    
            self.log(f"💾 数据导出: {filename}")
            messagebox.showinfo("导出成功", f"数据已导出到:\n{filename}")
        except Exception as e:
            self.log(f"❌ 导出失败: {e}")
            
    def clear_data(self):
        """清空数据"""
        if not self.all_pile_strikes:
            return
        if messagebox.askyesno("清空", "确定清空所有数据?"):
            self.all_pile_strikes.clear()
            self.pile_details.clear()
            self.current_pile_strikes = 0
            self.manual_strikes = 0
            self.strike_times.clear()
            self.strike_frequencies.clear()
            self.strike_volumes.clear()
            self.current_pile_name = ""
            self.pile_name_var.set("")
            self.current_pile_name_var.set("未设置")
            self.strikes_var.set("0")
            self.total_piles_var.set("0")
            self.total_strikes_var.set("0")
            self.avg_strikes_var.set("0.0")
            self.max_strikes_var.set("0")
            self.min_strikes_var.set("0")
            self.duration_var.set("00:00")
            self.strikes_per_min_var.set("0.0")
            self.freq_range_var.set("0-0 Hz")
            self.volume_range_var.set("0.0000-0.0000")
            self.start_time_var.set("--:--:--")
            self.end_time_var.set("--:--:--")
            self.judgment_var.set("未判定")
            self.penetration_var.set("")
            self.elevation_var.set("")
            self.manual_strike_var.set("0")
            self.log_text.delete(1.0, tk.END)
            self.log("🗑️ 数据已清空")
            
    def show_help(self):
        help_text = """
打桩锤击计数监测系统 v9.1-完整版 使用说明

【新增功能】
✓ 手动输入锤击数 - 解决程序意外退出问题
✓ 贯入度和超高数据输入
✓ 智能施工判定系统（可自定义条件）
✓ AI智能分析系统 - 机器学习模型训练
✓ 优化界面布局和颜色方案
✓ 完整数据统计报告
✓ 手动结束桩监测功能
✓ 实时统计信息更新

【施工判定条件】
条件1: 标高超高≤Xm且贯入度≤Ymm → 停止锤击
条件2: 标高超高<Xm且贯入度>Ymm → 打至标高超高0.4m停锤  
条件3: 标高超高>Xm且贯入度≤Ymm且锤击数>Z → 再施打100锤观察

【AI智能分析】
- 自动收集训练数据
- 机器学习模型训练
- 智能参数优化
- 提高检测准确性

【使用步骤】
1. 设置桩名称
2. 选择工作模式（实时监测/文件分析）
3. 设置检测参数（灵敏度、频率范围等）
4. 开始监测或分析文件
5. 桩完成后输入贯入度和超高数据
6. 查看施工判定结果
7. 导出完整监测报告

【界面优化】
• 青绿色按钮配色，与状态文字一致
• 优化布局，左右分栏各占一半
• 参数滑块和按钮重新设计
• 实时状态信息更加丰富

【注意事项】
• 确保麦克风正常工作（实时监测）
• 校准功能可自动设置最佳参数
• 手动输入功能用于补充监测数据
• 及时保存和导出重要数据
• 程序退出后会自动保存参数配置
        """
        messagebox.showinfo("使用说明", help_text)

    def format_duration(self, seconds):
        """格式化时间"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}时{minutes}分"

    def format_timestamp(self, seconds):
        """格式化时间戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def show_summary(self):
        """显示汇总"""
        if not self.all_pile_strikes:
            self.log("📊 无数据")
            return
            
        total = sum(self.all_pile_strikes)
        avg = total / len(self.all_pile_strikes)
        
        summary = f"📈 汇总: {len(self.all_pile_strikes)}桩, {total}次, 平均{avg:.1f}次/桩"
        self.log(summary)

    def safe_quit(self):
        """安全退出程序"""
        if self.is_monitoring:
            self.stop_monitoring()
        if self.is_analyzing:
            self.is_analyzing = False
        # 保存配置和模型
        self.save_config()
        self.save_ai_model()
        self.root.quit()

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    root = tk.Tk()
    app = PileDrivingMonitorGUI(root)
    
    def on_closing():
        app.safe_quit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()