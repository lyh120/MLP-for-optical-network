import json
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
import re
from requests.exceptions import ConnectionError

#### Qiu Qizhi 2023.8.31
#### Controlling code for ⅡⅥ systems

### Transceiver可以查询q值,查询ber，上下波，更改频率port，更改发射机光功率
class Transceiver:
    def __init__(self, ip, tp_id, tunnel_id):
        self.ip = ip
        # ip eg:"10.100.170.14"
        self.tp_id = tp_id
        # tp_id eg:"TDLC-1-1-L1-OTUC4"
        self.tunnel_id = tunnel_id
        # tunnel_id eg:"Tunnel-Site-1684124162727#Ne-1684139080695#LINECARD-1-1#PORT-1-1-C1-Site-1684124182011#Ne-1684139080708#LINECARD-1-1#PORT-1-1-C1"
        self.node_ref = tunnel_id[7:42]
        self.tp_ref = tunnel_id[7:65] + tp_id[9:11]
        self.friendly_name ='L' + str((int(tp_id[7])-1)*2 + int(tp_id[10]))
        self.headers = ''
        self.s = requests.session()
        self.ber_inst = 404
        self.q_inst = 404
        self.time = ''
        self.update_p_url = "http://10.100.170.13:8088/restconf/operations/otn-phy-topology:update-termination-point"
        self.update_url = "http://10.100.170.13:8088/restconf/operations/tunnel:update-tunnel"
        self.up_down_url = "http://10.100.170.13:8088/restconf/operations/tunnel:update-tunnel-sync"
        self.query_url = "http://10.100.170.13:8088/pmreport/monitor/latest"
        self.state_url ="http://10.100.170.13:8088/restconf/operations/nms:get-tunnel-paged"
        self.launchpower_url = "http://10.100.170.13:8088/restconf/operations/nms:get-route"
        self.history_url = "http://10.100.170.13:8088/pmreport/pm/history"
        self.get_headers()
        self.mux_port = self.get_mux_port()
        
    def get_headers(self):
        p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
        ret = self.s.post(p_url)
        # print(ret1.text)
        data = json.loads(ret.text)
        access_token = data['access_token']
        headers = {
            "Authorization": f"Bearer {access_token}"
                }
        self.headers = headers
    
    def get_mux_port(self):
        tunnel_body = \
        {
            "input": {
                "start-pos": 0,
                "topology-ref": "otn-phy-topology",
                "node-ref": self.node_ref,
                "tp-ref": self.tp_ref
            }
        }
        ret = self.s.post(self.state_url, headers=self.headers, json=tunnel_body)
        response = ret.json()
        port = response['output']['tunnel'][0]['properties']['property'][10]['value']
        self.mux_port = port
        # print("端口：" + port)
        return port

    def query_q(self):
        body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.tp_id
                            }
                        ],
                        "counter-type": "otn",
                        "group-name": self.tp_id,
                        "counters": [
                            "state.q_value"
                        ]
                    }
                ]
            }

        ret = self.s.post(self.query_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.query_url, headers=self.headers, json=body)
        
        response_data = ret.json()
        i=0
        # print(response_data['link'][0]['link-id'])
        while len(response_data[0]['counters'])==0:
            i = i + 1
            time.sleep(1)
            ret = self.s.post(self.query_url, headers=self.headers, json=body)
            response_data = ret.json()
            if i==40:
                print("查询次数超过40，Q值返回0，0，0，0")
                return 0,0,0,0
        # else:
        q_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
        q_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
        q_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
        q_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
        ts = response_data[0]['time-stamps'][0] / 1e3
        timeArray = time.localtime(ts)
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        # print()
        # print(time_now)
        # print(self.tunnel_id)
        # print('##############')
        # print('q_inst:'+q_inst)
        # print('q_max:'+q_max)
        # print('q_avg:'+q_avg)
        # print('q_min:'+q_min)
        print(self.friendly_name, 'q_avg:'+q_avg)
        self.q_inst = float(q_inst)
        self.time = time_now
        return float(q_inst), float(q_max), float(q_avg), float(q_min)
    
    def retry_on_connection_error(max_retries=3, delay=5):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except ConnectionError as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"连接错误，正在进行第{attempt+1}次重试...")
                        time.sleep(delay)
            return wrapper
        return decorator

    @retry_on_connection_error(max_retries=3, delay=5)
    def query_ber(self):
        body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.tp_id
                            }
                        ],
                        "counter-type": "otn",
                        "group-name": self.tp_id,
                        "counters": [
                            "state.pre_fec_ber",
                        ]
                    }
                ]
            }
        # time.sleep(2)
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.query_url, headers=self.headers, json=body, timeout=60)
                while(ret.status_code != 200):
                    time.sleep(1)
                    ret = self.s.post(self.query_url, headers=self.headers, json=body, timeout=60)
                response_data = ret.json()
                i = 0
                while len(response_data[0]['counters'])==0:
                    time.sleep(1)
                    i = i+1
                    ret = self.s.post(self.query_url, headers=self.headers, json=body, timeout=60)
                    response_data = ret.json()
                    if i==40:
                        print("查询次数超过40，BER值返回1，1，1，1")
                        return 1,1,1,1
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
        # else:
        # print(response_data['link'][0]['link-id'])
        ts = response_data[0]['time-stamps'][0] / 1e3
        timeArray = time.localtime(ts)
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        ber_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
        ber_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
        ber_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
        ber_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
        # if ber_inst == '1.000E+00':
        #     print(self.tunnel_id + " System Fail")
        # print()
        # print(time_now)
        # print(self.tunnel_id)
        # print('##############') 
        # print('ber_inst:'+ber_inst)
        # print('ber_max:'+ber_max)
        # print('ber_avg:'+ber_avg)
        # print('ber_min:'+ber_min)

        self.ber_inst = float(ber_inst)
        self.time = time_now
        return float(ber_inst), float(ber_max), float(ber_avg), float(ber_min)

    def up_tunnel(self):
        state = self.tunnel_state()
        if state =="implement":
            print("该波已部署")
            print()
            return True
        
        up = \
        {
            "input": {
                "admin-state": "up",
                "implement-state": "implement",
                "tunnel-id": self.tunnel_id
            }
        }
        ret = self.s.post(self.up_down_url, headers=self.headers, json=up)
        # print(ret.json())
        response = ret.json()
        if(response['output']['return-code']=='success'):
            print("UP 正在上业务 请等待生效")
        else:
            print("UP 上业务失败")
        print("正在上波...")
        
        time.sleep(60)
        for i in range(6):
            state = self.tunnel_state()
            if state=="doimplementing":
                print(self.friendly_name +"正在上波....")
                time.sleep(30)
            elif state=="partial-implement":
                print(self.friendly_name +"上波失败，再次尝试...")
                ret = self.s.post(self.up_down_url, headers=self.headers, json=up)
                time.sleep(30)
            elif state =="implement":
                port = self.get_mux_port()
                print(self.friendly_name +" 上波成功！端口："+port)
                print()
                return True
        if i==5:
            print(self.friendly_name +"操作失败请重试")
            return False

    def down_tunnel(self):
        state = self.tunnel_state()
        if state =="allocate":
            print(self.friendly_name +"该波并未部署")
            print()
            return True
        
        down = \
        {
            "input": {
                "admin-state": "down",
                "implement-state": "allocate",
                "tunnel-id":self.tunnel_id
            }
        }
        ret = self.s.post(self.up_down_url, headers=self.headers, json=down)
        # print(ret.json())
        print(self.friendly_name +"正在下波...")
        time.sleep(60)
        for i in range(6):
            state = self.tunnel_state()
            if state=="deimplementing":
                print(self.friendly_name +"正在下波....")
                time.sleep(30)
            elif state=="partial-implement":
                print(self.friendly_name +"下波失败，再次尝试...")
                ret = self.s.post(self.up_down_url, headers=self.headers, json=down)
                time.sleep(30)
            elif state =="allocate":
                print(self.friendly_name +" 下波成功！")
                print()
                return True
        if i==5:
            print(self.friendly_name +"操作失败请重试")
            return False
    
    def update_tunnel(self, new_mux_port):
        new_port = 65 - new_mux_port
        central_f = 191.2875 + new_port * 0.075
        low_f = (central_f - 0.0375)*1e6
        high_f = (central_f +0.0375)*1e6
        frequency =str(int(new_port)) + "-" + str(int(low_f)) + "," + str(int(high_f))

        update = \
        {
            "input": {
                "frequency": frequency,
                "tunnel-id": self.tunnel_id
            }
        }
        ret = self.s.post(self.update_url, headers=self.headers, json=update)

        return ret.status_code, ret.json()

    def update_tunnelpower(self, new_power):
        state = self.tunnel_state()
        if state != "implement":
            print(self.friendly_name+"并未上波，无法修改发射功率")
            print()
            return False
        
        update_p = \
    {
        "input": {
            "tps": [
                {
                    "tp-id": self.tp_ref,
                    "physical": {
                        "node-ref": self.node_ref,
                        "otu-line": {
                            "target-output-power": new_power
                        }
                    }
                }
            ]
        }
    }
        ret = self.s.post(self.update_p_url, headers=self.headers, json=update_p)
        time.sleep(10)
        for i in range(10):
            real_power = self.launch_power()
            if (real_power - new_power)>1e-5:
                print(self.friendly_name + "正在修改中....")
                time.sleep(5)
            elif (real_power - new_power)<1e-5:
                print("修改成功！")
                print(self.friendly_name + "发射机功率切换为：",new_power, " dBm")
                print()
                return True
        if i==9:
            print("操作失败请重试")
            return False
        
        # if ret.status_code == 200:
        #     print("发射机功率切换为：",new_power, "dBm")
        #     return ret.json()
        # else:
        #     print("功率切换失败！")
        #     print(ret.json())
        #     return ret.json()

    def tunnel_state(self):
        tunnel_body = \
        {
            "input": {
                "start-pos": 0,
                "topology-ref": "otn-phy-topology",
                "node-ref": self.node_ref,
                "tp-ref": self.tp_ref
            }
        }
        ret = self.s.post(self.state_url, headers=self.headers, json=tunnel_body)
        response = ret.json()
        state = response['output']['tunnel'][0]['implement-state']
        # print("状态：" + state)
        return state
    
    def launch_power(self):
        body = \
        {
            "input": {
                "topology-ref": "site-topology",
                "tunnel-ref": self.tunnel_id
            }
        }
        ret = self.s.post(self.launchpower_url, headers=self.headers, json=body)
        response = ret.json()
        
        for route_info in response['output']['route-info']:
            for route_sequence in route_info['primary']['route-sequence']:
                if 'tp-hop' in route_sequence:
                    phy_tp = route_sequence['tp-hop']['phy-tp']
                    if 'physical' in phy_tp and 'otu-line' in phy_tp['physical']:
                        otu_line = phy_tp['physical']['otu-line']
                        if 'target-output-power' and 'target-output-power-higher' in otu_line:
                            power = otu_line['target-output-power']
        return float(power)
    
    def switch_port(self, new_mux_port):
        new_port = 65 - new_mux_port
        central_f = 191.2875 + new_port * 0.075
        low_f = (central_f - 0.0375)*1e6
        high_f = (central_f +0.0375)*1e6
        frequency =str(int(new_port)) + "-" + str(int(low_f)) + "," + str(int(high_f))

        if(frequency==self.mux_port):
            state = self.tunnel_state()
            print("该收发机端口已经为:"+str(new_mux_port))
            print("状态为："+state)
            if state != 'implement':
                self.up_tunnel()
            elif state=='implement':
                print('********上波成功！切换频率到端口：'+ self.mux_port + ' ********')
            return 0
        
        state = self.tunnel_state()
        if state!='allocate':
            print("尝试切换中...")
            self.down_tunnel()
            
        self.update_tunnel(new_mux_port=new_mux_port)
        print("正在切换频率...")
        time.sleep(2)
        self.up_tunnel()

    def query_history_q(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")

        body =\
        {
        "start-timestamp": start,
        "end-timestamp": end,
        "history-interval": 0,
        "sample-export-interval": 0,
        "ip": self.ip,
        "name": self.tp_id,
        "pm-type": "otn"
        }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        q_values = []
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
      
            for counter in item['counter']:
                if counter['key'] == 'state.q_value':
                    q_value = float(counter['pm-four-compound-counter'][0]['value'])
                    q_values.append(q_value)
                    break
        return timestamps, q_values
    
    def query_history_ber(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")

        body =\
        {
        "start-timestamp": start,
        "end-timestamp": end,
        "history-interval": 0,
        "sample-export-interval": 0,
        "ip": self.ip,
        "name": self.tp_id,
        "pm-type": "otn"
        }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        ber_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            pre_fec_ber_avg = None

            # 遍历 counter 列表
            for counter in item['counter']:
                # 检查是否找到 'state.pre_fec_ber' 键
                if counter['key'] == 'state.pre_fec_ber':
                    # 在 pm-four-compound-counter 列表中查找 'avg' 值
                    for compound_counter in counter['pm-four-compound-counter']:
                        if compound_counter['key'] == 'avg':
                            pre_fec_ber_avg = compound_counter['value']
                            break
                    break
            
            # 如果找到了 pre_fec_ber_avg，则添加到列表中
            if pre_fec_ber_avg is not None:
                timestamps.append(timestamp)
                ber_values.append(float(pre_fec_ber_avg))
        
        return timestamps, ber_values
        
        
### BA 可调增益与增益斜率，增益范围7.0~31.0，斜率范围-2.0~0.0.
class BA:
    def __init__(self, ip, node_id):
        self.ip = ip
        self.node_id = node_id
        self.XC_id = "XC-"+node_id + "#LINECARD-1-1#PORT-1-1-SIG-" + node_id + "#LINECARD-1-1#PORT-1-1-LINE"
        self.headers = ''
        self.s = requests.session()
        self.get_headers()
        self.adjust_url = "http://10.100.170.13:8088/restconf/operations/otn-phy-topology:update-cross-connection"
        self.query_url = "http://10.100.170.13:8088/pmreport/monitor/latest"
        self.history_url = "http://10.100.170.13:8088/pmreport/pm/history"
        self.alarm_log_url = "http://10.100.170.13:8088/restconf/operations/alarm:get-history-alarms"
        self.alarm_url = "http://10.100.170.13:8088/restconf/operations/alarm:get-current-alarms"

    def set_gain(self, gain_dB):
        if (gain_dB > 31) or (gain_dB < 7):
            print("BA增益范围为7.0~31.0！请重新设置")
            return 0
        
        if gain_dB < 22:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:LOW_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        else:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:HIGH_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=g_body)
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！BA增益修改为:", gain_dB, "dB")
                    return True
                else:
                    print("增益修改失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
    def set_tilt(self, gain_tilt):
        if (gain_tilt > 0) or (gain_tilt < -2):
            print("BA增益斜率范围为-2.0~0.0！请重新设置")
            return 0

        t_body = \
        {
            "input": {
                "xcs": [
                    {
                        "node-id": self.node_id,
                        "cross-connection-id": self.XC_id,
                        "amplifier": {
                            "target-gain-tilt": str(np.round(gain_tilt,1)),
                        }
                    }
                ]
            }
        }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=t_body)
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！BA增益斜率为:", gain_tilt)
                    return True
                else:
                    print("修改斜率失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避

    def query_input_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.node_id + "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": "AMPLIFIER-1-1-BA",
                        "counters": [
                        "state.input_power_total" 
                        ]
                    }
                ]
            }
        # time.sleep(1)
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        if ret.status_code == 200:
            response_data = ret.json()
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
            # else:
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " BA input power")
            # print('##############')
            # print('input power inst:'+p_inst)
            # print('input power max:'+p_max)
            # print('input power avg:'+p_avg)
            # print('input power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"BA性能查询报错 错误码: {ret.status_code}")
        
    def query_output_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.node_id+ "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": "AMPLIFIER-1-1-BA",
                        "counters": [
                        "state.output_power_total" 
                        ]
                    }
                ]
            }
        
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        if ret.status_code == 200:
            response_data = ret.json()
            # print(response_data['link'][0]['link-id'])
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
            # else:
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " BA output power")
            # print('##############')
            # print('output power inst:'+p_inst)
            # print('output power max:'+p_max)
            # print('output power avg:'+p_avg)
            # print('output power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"BA性能查询报错 错误码: {ret.status_code}")
            
    def query_real_gain(self):
        _, _, output, _ = self.query_output_power()
        _, _, input,_ = self.query_input_power()
        gain = round((output-input), 1)
        print(self.ip + " BA real gain:", gain)
        return gain

    def get_headers(self):
        p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
        ret = self.s.post(p_url)
        # print(ret1.text)
        data = json.loads(ret.text)
        access_token = data['access_token']
        headers = {
            "Authorization": f"Bearer {access_token}"
                }
        self.headers = headers

    def query_history_input_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
        
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": "AMPLIFIER-1-1-BA",
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        input_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.input_power_total':
                    input_power = float(counter['pm-four-compound-counter'][0]['value'])
                    input_power_values.append(input_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                input_power_values.append(None)
        return timestamps, input_power_values
    
    def query_history_output_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": "AMPLIFIER-1-1-BA",
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        output_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.output_power_total':
                    output_power = float(counter['pm-four-compound-counter'][0]['value'])
                    output_power_values.append(output_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                output_power_values.append(None)
        return timestamps, output_power_values
    
    def query_alarm_log(self):
    # 查询网管处OA的历史告警
        body = \
            {
            "input": {
                "start-pos": 0,
                "how-many": 20,
                "sort-infos": [
                {
                    "ascending": False,
                    "sort-name": "creation-time"
                }
                ],
                "object-info": {
                "object-id": self.node_id + "#LINECARD-1-1#PORT-1-1-LINE",
                "object-type": "tp"
                }
            }
            }
        ret = self.s.post(self.alarm_log_url, headers=self.headers, data=json.dumps(body))
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.alarm_log_url, headers=self.headers, data=json.dumps(body))
        response_data = ret.json()
        alarms = [
        {
            "resource_ref": alarm.get("resource-ref"),
            "alarm_type_id": alarm.get("alarm-type-id"),
            "nml_key_name": alarm.get("nml-key-name"),
            "equipment_ref": alarm.get("equipment-ref"),
            "alarm_group": alarm.get("alarm-group"),
            "creation_received_time": alarm.get("creation-received-time"),
            "archive_time": alarm.get("archive-time"),
            "alarm_id": alarm.get("alarm-id"),
            "sa": alarm.get("sa"),
            "clear_time": alarm.get("clear-time"),
            "alarm_text": alarm.get("alarm-text"),
            "archive_type": alarm.get("archive-type"),
            "creation_time": alarm.get("creation-time"),
            "serverity": alarm.get("serverity"),
            "nml_key": alarm.get("nml-key"),
            "ne_id": alarm.get("ne-id"),
            "alarm_index": alarm.get("alarm-index")
        }
        for alarm in response_data.get("output", {}).get("alarm", [])
        ]
        return alarms

    def query_alarm_current(self):
    # 查询网管处OA的历史告警
        body = \
        {
        "input": {
            "start-pos": 0,
            "how-many": 20,
            "sort-infos": [
            {
                "ascending": False,
                "sort-name": "creation-time"
            }
            ],
            "object-info": {
            "object-id": self.node_id + "#LINECARD-1-1#PORT-1-1-LINE",
            "object-type": "tp"
            }
        }
        }
        ret = self.s.post(self.alarm_url, headers=self.headers, data=json.dumps(body))
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.alarm_url, headers=self.headers, data=json.dumps(body))
        response_data = ret.json()
        alarms = []
        if 'output' in response_data and 'alarm' in response_data['output']:
            for alarm in response_data['output']['alarm']:
                alarm_info = {
                    "alarm_id": alarm.get("alarm-id"),
                    "resource_ref": alarm.get("resource-ref"),
                    "alarm_type_id": alarm.get("alarm-type-id"),
                    "nml_key_name": alarm.get("nml-key-name"),
                    "equipment_ref": alarm.get("equipment-ref"),
                    "alarm_group": alarm.get("alarm-group"),
                    "creation_received_time": alarm.get("creation-received-time"),
                    "sa": alarm.get("sa"),
                    "alarm_text": alarm.get("alarm-text"),
                    "creation_time": alarm.get("creation-time"),
                    "serverity": alarm.get("serverity"),
                    "nml_key": alarm.get("nml-key"),
                    "ne_id": alarm.get("ne-id"),
                    "alarm_index": alarm.get("alarm-index")
                }
                alarms.append(alarm_info)
        return alarms

### PA 可调增益与增益斜率，增益范围16.0~32.0，斜率范围-2.0~0.0,前置光衰0.0~30.0
class PA:
    def __init__(self, ip, node_id):
        self.ip = ip
        self.node_id = node_id
        self.XC_id = "XC-"+node_id + "#LINECARD-1-1#PORT-1-1-LINE-" + node_id + "#LINECARD-1-1#PORT-1-1-SIG"
        self.headers = ''
        self.s = requests.session()
        self.get_headers()
        self.adjust_url = "http://10.100.170.13:8088/restconf/operations/otn-phy-topology:update-cross-connection"
        self.query_url = "http://10.100.170.13:8088/pmreport/monitor/latest"
        self.history_url = "http://10.100.170.13:8088/pmreport/pm/history"
        
    def set_gain(self, gain_dB):
        if (gain_dB > 32) or (gain_dB < 16):
            print("PA增益范围为16.0~32.0！请重新设置")
            return 0
        if gain_dB <= 22:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:LOW_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        else:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:HIGH_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=g_body, timeout=(10,60))
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！PA增益为:", gain_dB, "dB")
                    return True
                else:
                    print("修改失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
    def set_tilt(self, gain_tilt):
        if (gain_tilt > 0) or (gain_tilt < -2):
            print("PA增益斜率范围为-2.0~0.0！请重新设置")
            return 0

        t_body = \
        {
            "input": {
                "xcs": [
                    {
                        "node-id": self.node_id,
                        "cross-connection-id": self.XC_id,
                        "amplifier": {
                            "target-gain-tilt": str(gain_tilt),
                        }
                    }
                ]
            }
        }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=t_body, timeout=(10,60))
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！PA增益斜率修改为:", gain_tilt)
                    return True
                else:
                    print("增益修改失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避

    def atten(self, atten):
        if (atten > 30) or (atten < 0):
            print("PA前置VOA衰耗范围为0.0~30.0 dB！请重新设置")
            return 0

        a_body = \
        {
            "input": {
                "xcs": [
                    {
                        "node-id": self.node_id,
                        "cross-connection-id": self.XC_id,
                        "amplifier": {
                            "target-attenuation": str(atten),
                        }
                    }
                ]
            }
        }
        ret = self.s.post(self.adjust_url, headers=self.headers, json=a_body)
        time.sleep(0.8)
        if ret.status_code == 200:
            print("修改成功！PA前置VOA衰耗为:", atten, "dB")

    def query_input_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.node_id+ "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": "AMPLIFIER-1-1-PA",
                        "counters": [
                        "state.input_power_total" 
                        ]
                    }
                ]
            }
        # time.sleep(1)
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        if ret.status_code == 200:
            response_data = ret.json()
            # print(response_data['link'][0]['link-id'])
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
            # else:
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " PA input power")
            # print('##############')
            # print('input power inst:'+p_inst)
            # print('input power max:'+p_max)
            # print('input power avg:'+p_avg)
            # print('input power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"PA性能查询报错 错误码: {ret.status_code}")
        
    def query_output_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id": self.node_id+ "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": "AMPLIFIER-1-1-PA",
                        "counters": [
                        "state.output_power_total" 
                        ]
                    }
                ]
            }
        # time.sleep(1)
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        if ret.status_code == 200:
            response_data = ret.json()
            # print(response_data['link'][0]['link-id'])
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " PA output power")
            # print('##############')
            # print('output power inst:'+p_inst)
            # print('output power max:'+p_max)
            # print('output power avg:'+p_avg)
            # print('output power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"PA性能查询报错 错误码: {ret.status_code}")
            
    def query_real_gain(self):
        _, _, output, _ = self.query_output_power()
        _, _, input,_ = self.query_input_power()
        gain = round((output-input), 1)
        print(self.ip + " PA real gain:", gain)
        return gain
    
    def get_headers(self):
        p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
        ret = self.s.post(p_url)
        # print(ret1.text)
        data = json.loads(ret.text)
        access_token = data['access_token']
        headers = {
            "Authorization": f"Bearer {access_token}"
                }
        self.headers = headers
    
    def query_history_input_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):

        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)

        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
    
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": "AMPLIFIER-1-1-PA",
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        input_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.input_power_total':
                    input_power = float(counter['pm-four-compound-counter'][0]['value'])
                    input_power_values.append(input_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                input_power_values.append(None)
        return timestamps, input_power_values
    
    def query_history_output_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": "AMPLIFIER-1-1-PA",
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        output_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.output_power_total':
                    output_power = float(counter['pm-four-compound-counter'][0]['value'])
                    output_power_values.append(output_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                output_power_values.append(None)
        return timestamps, output_power_values
    
### ILA 可调增益与增益斜率，增益范围16.0~32.0，斜率范围-2.0~0.0,前置光衰0.0~30.0
class ILA:
    def __init__(self, ip, node_id, direction):
        self.ip = ip
        self.node_id = node_id
        self.A2B_id = "XC-"+node_id + "#LINECARD-1-1#PORT-1-1-LINEA-" + node_id + "#LINECARD-1-1#PORT-1-1-LINEB"
        self.B2A_id = "XC-"+node_id + "#LINECARD-1-1#PORT-1-1-LINEB-" + node_id + "#LINECARD-1-1#PORT-1-1-LINEA"
        self.headers = ''
        self.s = requests.session()
        self.get_headers()
        self.adjust_url = "http://10.100.170.13:8088/restconf/operations/otn-phy-topology:update-cross-connection"
        self.query_url = "http://10.100.170.13:8088/pmreport/monitor/latest"
        if direction == 'A2B':
            self.XC_id = self.A2B_id
            self.group_name = "AMPLIFIER-1-1-ILAA"
        elif direction=='B2A':
            self.XC_id = self.B2A_id
            self.group_name = "AMPLIFIER-1-1-ILAB"
        self.history_url = "http://10.100.170.13:8088/pmreport/pm/history"
        
    def set_gain(self, gain_dB):
        if (gain_dB > 32) or (gain_dB < 16):
            print("ILA增益范围为16.0~32.0！请重新设置")
            return 0
        if gain_dB <= 22:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:LOW_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        else:
            g_body = \
            {
                "input": {
                    "xcs": [
                        {
                            "node-id": self.node_id,
                            "cross-connection-id": self.XC_id,
                            "amplifier": {
                                "target-gain": str(np.round(gain_dB,1)),
                                "gain-range": "common-otn-types:HIGH_GAIN_RANGE"
                            }
                        }
                    ]
                }
            }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=g_body)
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！ILA增益为:", gain_dB, "dB")
                    return True
                else:
                    print("修改失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
    def set_tilt(self, gain_tilt):
        if (gain_tilt > 0) or (gain_tilt < -2):
            print("ILA增益斜率范围为-2.0~0.0！请重新设置")
            return 0
        t_body = \
        {
            "input": {
                "xcs": [
                    {
                        "node-id": self.node_id,
                        "cross-connection-id": self.XC_id,
                        "amplifier": {
                            "target-gain-tilt": str(gain_tilt),
                        }
                    }
                ]
            }
        }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=t_body)
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！ILA增益斜率为:", gain_tilt)
                    return True
                else:
                    print("修改失败")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
        
    def atten(self, atten):
        if (atten > 30) or (atten < 0):
            print("ILA前置VOA衰耗范围为0.0~30.0 dB！请重新设置")
            return 0
        a_body = \
        {
            "input": {
                "xcs": [
                    {
                        "node-id": self.node_id,
                        "cross-connection-id": self.XC_id,
                        "amplifier": {
                            "target-attenuation": str(atten),
                        }
                    }
                ]
            }
        }
        for attempt in range(3):  # 尝试3次
            try:
                ret = self.s.post(self.adjust_url, headers=self.headers, json=a_body)
                time.sleep(0.8)
                if ret.status_code == 200:
                    print("修改成功！ILA前置VOA衰耗为:", atten, "dB")
                    return
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # 如果是最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
    def query_input_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id":self.node_id + "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": self.group_name,
                        "counters": [
                        "state.input_power_total" 
                        ]
                    }
                ]
            }

        # time.sleep(1)
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        if ret.status_code == 200:
            response_data = ret.json()
            # print(response_data['link'][0]['link-id'])
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
        # else:
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " ILA input power")
            # print('##############')
            # print('input power inst:'+p_inst)
            # print('input power max:'+p_max)
            # print('input power avg:'+p_avg)
            # print('input power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"ILA性能查询报错 错误码: {ret.status_code}")
    
    def query_output_power(self):
        q_body = \
            {
                "items": [
                    {
                        "tps": [
                            {
                                "ip": self.ip,
                                "tp-id":self.node_id + "#LINECARD-1-1"
                            }
                        ],
                        "counter-type": "amplifier",
                        "group-name": self.group_name,
                        "counters": [
                        "state.output_power_total" 
                        ]
                    }
                ]
            }
        # time.sleep(1)
        ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
        i = 0
        if ret.status_code == 200:
            response_data = ret.json()
            # print(response_data['link'][0]['link-id'])
            while len(response_data[0]['counters'])==0:
                time.sleep(1)
                i = i + 1
                ret = self.s.post(self.query_url, headers=self.headers, json=q_body)
                response_data = ret.json()
                if(i==10):
                    return False
        # else:
            p_inst = response_data[0]['counters'][0]['data'][0]['data'][0][0]
            p_max = response_data[0]['counters'][0]['data'][0]['data'][0][1]
            p_avg = response_data[0]['counters'][0]['data'][0]['data'][0][2]
            p_min = response_data[0]['counters'][0]['data'][0]['data'][0][3]
            ts = response_data[0]['time-stamps'][0] / 1e3
            timeArray = time.localtime(ts)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            # print()
            # print(time_now)
            # print(self.ip + " ILA output power")
            # print('##############')
            # print('output power inst:'+p_inst)
            # print('output power max:'+p_max)
            # print('output power avg:'+p_avg)
            # print('output power min:'+p_min)
            return float(p_inst), float(p_max), float(p_avg), float(p_min)
        else:
            print(f"ILA性能查询报错 错误码: {ret.status_code}")
            
    def query_real_gain(self):
        if not self.query_output_power():
                return False
        _, _, output, _ = self.query_output_power()
        _, _, input,_ = self.query_input_power()
        gain = round((output-input), 1)
        print(self.group_name + " real gain:", gain)
        return gain
    
    def get_headers(self):
        p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
        ret = self.s.post(p_url)
        # print(ret1.text)
        data = json.loads(ret.text)
        access_token = data['access_token']
        headers = {
            "Authorization": f"Bearer {access_token}"
                }
        self.headers = headers
    
    def query_history_input_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):

        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
        
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": self.group_name,
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        input_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.input_power_total':
                    input_power = float(counter['pm-four-compound-counter'][0]['value'])
                    input_power_values.append(input_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                input_power_values.append(None)
        return timestamps, input_power_values
    
    def query_history_output_power(self, start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00"):
        start = time_to_timestamp(start_time)
        end = time_to_timestamp(end_time)
        if end <= start:
            raise ValueError("结束时间必须在开始时间之后")
        
        if end - start > 3 * 60 * 60 * 1000:
            raise ValueError("时间间隔不能超过3小时")
        
        body =\
            {
            "start-timestamp": start,
            "end-timestamp": end,
            "history-interval": 0,
            "sample-export-interval": 0,
            "ip": self.ip,
            "name": "AMPLIFIER-1-1-PA",
            "pm-type": "amplifier"
            }
        ret = self.s.post(self.history_url, headers=self.headers, json=body)
        while(ret.status_code != 200):
            time.sleep(1)
            ret = self.s.post(self.history_url, headers=self.headers, json=body)
        response_data = ret.json()
        
        timestamps = []
        output_power_values = []
        
        for item in response_data['pm-data']:
            timestamp = item['time-stamp']
            beijing_time = timestamp_to_beijing_time(timestamp)
            timestamps.append(beijing_time)
            
            for counter in item['counter']:
                if counter['key'] == 'state.output_power_total':
                    output_power = float(counter['pm-four-compound-counter'][0]['value'])
                    output_power_values.append(output_power)
                    break
            else:
                # 如果没有找到 state.input_power_total，添加 None 或者其他标记值
                output_power_values.append(None)
        return timestamps, output_power_values
    
class OCM:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.headers = ''
        self.s = requests.session()
        self.get_headers()
        self.ocm_url = "http://10.100.170.13:8088/pmreport/ocm/latest"
    
    def get_headers(self):
        p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
        ret = self.s.post(p_url)
        # print(ret1.text)
        data = json.loads(ret.text)
        access_token = data['access_token']
        headers = {
            "Authorization": f"Bearer {access_token}"
                }
        self.headers = headers
    
    def get(self):
        body = \
        {
        "ip": self.ip,
        "name": "PORT-1-1-"+self.port
        }
        ret = self.s.post(self.ocm_url, headers=self.headers, json=body)
        if ret.status_code == 200:
            response_data = ret.json()
            while len(response_data["ocm-channel-datas"])==0:
                time.sleep(2)
                ret = self.s.post(self.ocm_url, headers=self.headers, json=body)
                response_data = ret.json()
            data = response_data["ocm-channel-datas"][0]["ocm-channel"]
            psd = [entry['psd'] for entry in data]
            f = [(entry['lower-frequency'] + entry['upper-frequency']) / 2 for entry in data]
            return psd, f
        else:
            print(f"OCM查询报错 错误码: {ret.status_code}")
    
    def show(self):
        psd, f = self.get()
        fig, ax = plt.subplots()
        line, = ax.plot(f, psd, '-o')
        
        ax.set_title(self.ip + '-' + self.port + '-OCM')
        ax.set_xlabel('Frequency[Hz]', fontsize = 12)
        ax.set_ylabel('PSD', fontsize=12)
        plt.tick_params(labelsize=12)
        # plt.ylim(-4,7.5)
        plt.show()
        
    def save(self, name):
        psd, f = self.get()
        fig, ax = plt.subplots()
        line, = ax.plot(f, psd)
        ax.set_title(self.ip + '-' + self.port + '-OCM')
        ax.set_xlabel('Frequency[Hz]', fontsize = 12)
        ax.set_ylabel('PSD', fontsize=12)
        plt.tick_params(labelsize=12)
        plt.ylim(-4,7.5)
        plt.savefig('./Fig/'+name+'.png')

    def channel_power(self):
        psd, f = self.get()
        f = np.array(f)
        psd = np.array(psd)
        power_mean = np.zeros(64)
        channel_occ_ind = np.ones(64)
        channel_occ_ind[0:3] = -80
        channel_occ_ind[52:] = -80
        for index_of_channel in range(64):
            if channel_occ_ind[index_of_channel] != -80 :
                center_fre = 196.0875 - (index_of_channel) * 0.075
                high_freq = (center_fre+0.07/2)*1e6
                low_freq = (center_fre-0.07/2)*1e6
                wsFreq_channel_sig_freq_index =np.where((f<high_freq)& (f>low_freq))
                power_mean[index_of_channel] = np.mean(psd[wsFreq_channel_sig_freq_index])-10*np.log10(50/63.91)
        return power_mean

def ini_OCMs():
    """
    Tx ─── mux ─── edfa1(OCM1) ─── fiber[0] ─── edfa2(OCM2) ─── fiber[1] ─── edfa3(OCM3)  ──┐
                                                                                            │
                                                                                            fiber[2]
                                                                                            │
    Rx ─── mux ─── (OCM6)edfa6 ─── fiber[4] ─── (OCM5)edfa5 ─── fiber[3] ─── (OCM4)edfa4  ──┘
    """
    ip_sh = "10.100.170.10"
    ip_jx = "10.100.170.11"
    ip_hz = "10.100.170.12"
    OCM1 = OCM(ip=ip_sh, port="SIG")
    OCM2 = OCM(ip=ip_jx, port="LINEA")
    OCM3 = OCM(ip=ip_hz, port="LINE")
    OCM4 = OCM(ip=ip_hz, port="SIG")
    OCM5 = OCM(ip=ip_jx, port="LINEB")
    OCM6 = OCM(ip=ip_sh, port="LINE")
    
    return OCM1, OCM2, OCM3, OCM4, OCM5, OCM6

def query_setting():
    p_url = "http://10.100.170.13:8088/oauth2/token?username=U0pUVV9aaHVnZQ==&password=MTMxMjI4&client_id=password-dci&client_secret=dciWorld&scope=all&grant_type=password"
    s = requests.session()
    ret = s.post(p_url)
    # print(ret1.text)
    data = json.loads(ret.text)
    access_token = data['access_token']
    headers = {
        "Authorization": f"Bearer {access_token}"
            }
    
    q_url = "http://10.100.170.13:8088/restconf/operations/nms:get-route"
    body = \
    {
    "input": {
        "topology-ref": "site-topology",
        "tunnel-ref": "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-1#PORT-1-1-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-1#PORT-1-1-C1"
    }
    }
    ret = s.post(q_url, headers=headers, json=body)
    response = ret.json()
    xc = response["output"]["route-info"][0]["primary"]["cross-connections"]
    name = []
    gain = []
    tilt = []
    for i in range (6):
        name.append(xc[i+4]["description"])
        gain.append(xc[i+4]["amplifier"]["target-gain"])
        tilt.append(xc[i+4]["amplifier"]["target-gain-tilt"])
        # print()
        # print(name[i])
        # print("Gain:", gain[i], "dB")
        # print("Gain tilt:", tilt[i])
    new_order = [0, 2, 5, 4, 3, 1]
    new_gain = [gain[i] for i in new_order]
    new_tilt = [tilt[i] for i in new_order]
    return new_gain, new_tilt

def wave(wave_occ, L):
    state = []
    for i in range(len(wave_occ)):
        state.append(L[i].tunnel_state())
        time.sleep(0.8)
        if(wave_occ[i]==0):
            if(state[i]=='allocate'):
                continue
            else:
                print(f'Down L{i+1}')
                L[i].down_tunnel()
                
        elif(wave_occ[i]==1):
            if(state[i]=='implement'):
                continue
            else:
                print(f'Up L{i+1}')
                L[i].up_tunnel()

def ini_TRx():
    L_ip = "10.100.170.14"
    L1_tp_id = "TDLC-1-1-L1-OTUC2"
    L1_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-1#PORT-1-1-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-1#PORT-1-1-C1"

    L2_tp_id = "TDLC-1-1-L2-OTUC2"
    L2_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-1#PORT-1-1-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-1#PORT-1-1-C5"

    L3_tp_id = "TDLC-1-2-L1-OTUC2"
    L3_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-2#PORT-1-2-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-2#PORT-1-2-C1"

    L4_tp_id = "TDLC-1-2-L2-OTUC2"
    L4_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-2#PORT-1-2-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-2#PORT-1-2-C5"

    L5_tp_id = "TDLC-1-3-L1-OTUC2"
    L5_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-3#PORT-1-3-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-3#PORT-1-3-C1"

    L6_tp_id = "TDLC-1-3-L2-OTUC2"
    L6_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-3#PORT-1-3-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-3#PORT-1-3-C5"



    L1 = Transceiver(ip=L_ip, tp_id=L1_tp_id,tunnel_id=L1_tunnel_id)
    # L1.update_tunnelpower(-3)
    L2 = Transceiver(ip=L_ip, tp_id=L2_tp_id,tunnel_id=L2_tunnel_id)
    # L2.update_tunnelpower(-2.6)
    L3 = Transceiver(ip=L_ip, tp_id=L3_tp_id,tunnel_id=L3_tunnel_id)
    # L3.update_tunnelpower(-2.9)
    L4 = Transceiver(ip=L_ip, tp_id=L4_tp_id,tunnel_id=L4_tunnel_id)
    # L4.update_tunnelpower(-2.7)
    L5 = Transceiver(ip=L_ip, tp_id=L5_tp_id,tunnel_id=L5_tunnel_id)
    # L5.update_tunnelpower(-2.6)
    L6 = Transceiver(ip=L_ip, tp_id=L6_tp_id,tunnel_id=L6_tunnel_id)
    # L6.update_tunnelpower(-1.9)
    return L1,L2,L3,L4,L5,L6

def ini_TencentOA():
    node_id_15 = "Site-1691561426178#Ne-1691561630904"
    node_id_11 = "Site-1691561490497#Ne-1691561630919"
    node_id_12 = "Site-1691561501945#Ne-1691561630930"
    
    ip10 = "10.100.170.10"
    ip11 = "10.100.170.11"
    ip12 = "10.100.170.12"
    # edfa = EDFA_BA('COM5', '测试的EDFA', logger=logger, flag_broadcast=False, slot=2)
    b15 = BA(ip = ip10, node_id=node_id_15)
    p15 = PA(ip = ip10, node_id=node_id_15)
    iA2B = ILA(ip = ip11, node_id=node_id_11, direction='A2B')
    iB2A = ILA(ip = ip11, node_id=node_id_11, direction='B2A')
    b12 = BA(ip = ip12, node_id=node_id_12)
    p12 = PA(ip=ip12, node_id=node_id_12)
    return b15, p15, iA2B, iB2A, b12, p12

def time_to_timestamp(time="2024-08-23 15:54:16"):
    # 创建北京时区对象
    beijing_tz = pytz.timezone('Asia/Shanghai')
    # 使用正则表达式匹配时间格式
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})'
    match = re.match(pattern, time)
    if not match:
        raise ValueError("Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS'.")
    # 提取时间组件并转换为整数
    year, month, day, hour, minute, second = map(int, match.groups())
    # 创建datetime对象
    dt = datetime(year, month, day, hour, minute, second)
    # 为datetime对象添加时区信息
    dt_with_tz = beijing_tz.localize(dt)
    # 转换为时间戳（毫秒级）
    timestamp = int(dt_with_tz.timestamp() * 1000)
    return timestamp

def timestamp_to_beijing_time(timestamp):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).astimezone(beijing_tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# def check_alarm(OA_list, alarm_list):
#     if len(alarm_list)==0:
#         return "No alarm in the system"

#     target_text = "OSC_INPUT_LOS;OSC输入无光"
#     for alarm in alarm_list:
#         if 'alarm_text' in alarm and target_text in alarm['alarm_text']:
#             return "No OSC signal received"

#     for i in range(len(OA_list)):
#         oa = OA_list[i]
#         if not oa.query_real_gain():
#             return f"EDFA{i+1} out of control!"
#         if oa.query_real_gain()<5:
#             return f"EDFA{i+1} failure!"
    
#     for j in range(len(OA_list)):
#         oa = OA_list[j]
#         if oa.query_input_power()<-40:
#             match i:
#                 case 0:
#                     return "Transmitter Down!"
#                 case 1:
#                     return "Shanghai to Jiaxing fiber optic cable damage!"
#                 case 2:
#                     return "Jiaxing to Hangzhou fiber optic cable damage!"
#                 case 4:
#                     return "Hangzhou to Jiaxing fiber optic cable damage!"
#                 case 5:
#                     return "Jiaxing to Shanghai fiber optic cable damage!"
#             return f"EDFA{i+1} failure!"

if __name__ == '__main__':
    # 定义业务
    L_ip = "10.100.170.14"
    L1_tp_id = "TDLC-1-1-L1-OTUC2"
    L1_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-1#PORT-1-1-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-1#PORT-1-1-C1"

    L2_tp_id = "TDLC-1-1-L2-OTUC2"
    L2_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-1#PORT-1-1-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-1#PORT-1-1-C5"

    L3_tp_id = "TDLC-1-2-L1-OTUC2"
    L3_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-2#PORT-1-2-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-2#PORT-1-2-C1"

    L4_tp_id = "TDLC-1-2-L2-OTUC2"
    L4_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-2#PORT-1-2-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-2#PORT-1-2-C5"

    L5_tp_id = "TDLC-1-3-L1-OTUC2"
    L5_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-3#PORT-1-3-C1-Site-1691561501945#Ne-1691561720395#LINECARD-1-3#PORT-1-3-C1"

    L6_tp_id = "TDLC-1-3-L2-OTUC2"
    L6_tunnel_id = "Tunnel-Site-1691561426178#Ne-1691561720381#LINECARD-1-3#PORT-1-3-C5-Site-1691561501945#Ne-1691561720395#LINECARD-1-3#PORT-1-3-C5"


    ip15 = "10.100.170.15"
    ip16 = "10.100.170.16"
    ip17 = "10.100.170.17"
    node_id_15 = "Site-1691561426178#Ne-1691561630904"
    node_id_17 = "Site-1691561501945#Ne-1691561630930"
    node_id_16 = "Site-1691561490497#Ne-1691561630919"
    
    # L1, L2, L3, L4, L5, L6 = ini_TRx()
    # time, q = L1.query_history_q(start_time="2024-8-23 11:10:00", end_time="2024-8-23 11:30:00")
    OCM1, OCM2, OCM3, OCM4, OCM5, OCM6 = ini_OCMs()
    OCM_list = [OCM1, OCM2, OCM3, OCM4, OCM5, OCM6]
    for ocm in OCM_list:
        ocm.show()
    
    EDFA1, EDFA6, EDFA2, EDFA5, EDFA3, EDFA4 = ini_TencentOA()
    OA_list = [EDFA1, EDFA2, EDFA3, EDFA4, EDFA5, EDFA6]
    print(EDFA1.query_input_power())
    print(EDFA1.query_output_power())
    # tt, p = b15.query_history_input_power(start_time="2024-08-23 11:23:30", end_time="2024-08-23 11:24:00")
    # real_gain = iB2A.query_real_gain()
    # a = b15.query_alarm_current()
    # alarm = EDFA1.query_alarm_log()
    # flag = check_alarm(OA_list, alarm)
    # print(flag)





    # L1 = Transceiver(ip=L_ip, tp_id=L1_tp_id,tunnel_id=L1_tunnel_id)
    # L1.update_tunnelpower(-2)
    # L2 = Transceiver(ip=L_ip, tp_id=L2_tp_id,tunnel_id=L2_tunnel_id)
    # L2.update_tunnelpower(-1.9)
    # L3 = Transceiver(ip=L_ip, tp_id=L3_tp_id,tunnel_id=L3_tunnel_id)
    # L3.update_tunnelpower(-1.8)
    # L4 = Transceiver(ip=L_ip, tp_id=L4_tp_id,tunnel_id=L4_tunnel_id)
    # L4.update_tunnelpower(-1.7)
    # L5 = Transceiver(ip=L_ip, tp_id=L5_tp_id,tunnel_id=L5_tunnel_id)
    # L5.update_tunnelpower(-1.6)
    # L6 = Transceiver(ip=L_ip, tp_id=L6_tp_id,tunnel_id=L6_tunnel_id)
    # L6.update_tunnelpower(-1.5)