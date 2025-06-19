import sys
import os

# 将上层目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys

from utilities.utils import *
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import(
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from typing import Optional, Type
from sklearn.metrics import mean_squared_error
import numpy as np
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
import langchain
#设置代理
import os

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# %%
#系统性能查询 工具定义,收发机的个数暂定为4个

class custom_query_Q(BaseTool):
    name = "query_Q"
    description = '''
        Returns: Four float values representing the Q-factors of four different transceivers.
        Note: Query the Q-factors of four diferent tranceivers and return the values.
        '''
    return_direct: bool = False

    def _run(
        self,
        tool_input: str='',
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    )->str:
        '''use the tool'''
        ## 多次查询平均，平均次数可调
        return query_parallel()

    async def _arun(self,
                    tool_input: str='',
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                    **kwargs
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(run_manager=run_manager.get_sync())
    
    # def run(self, tool_input: Optional[str] = None, **kwargs) -> str:  # 覆盖 run 方法，提供默认值
    #     return self._run(tool_input=tool_input)
    
# %%
###### 单个EDFA调整
## EDFA Input
class set_EDFA_Input(BaseModel):
    gain: float = Field(description="the gain value set for EDFA")
    tilt: float = Field(description="the tilt value set for EDFA")

## EDFA 1
class custom_set_EDFA_1(BaseTool):
    name = "set_EDFA_1"
    description = '''
        Args: a float value representing the gain set for EDFA1.a float value representing the tilt set for EDFA1.
        Returns: The actual gain setting and tilt setting values of EDFA1.
        Note: Set the gain and tilt for EDFA1 and returns the actual gain setting and tilt setting values of EDFA1.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA1(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())

## EDFA 2
class custom_set_EDFA_2(BaseTool):
    name = "set_EDFA_2"
    description = '''
        Args: a float value representing the gain set for EDFA2.a float value representing the tilt set for EDFA2.
        Returns: The actual gain setting and tilt setting values of EDFA2.
        Note: Set the gain and tilt for EDFA2
        and returns the actual gain setting and tilt setting values of EDFA2.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA2(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())

## EDFA 3
class custom_set_EDFA_3(BaseTool):
    name = "set_EDFA_3"
    description = '''
        Args: a float value representing the gain set for EDFA3.a float value representing the tilt set for EDFA3.
        Returns: The actual gain setting and tilt setting values of EDFA3.
        Note: Set the gain and tilt for EDFA3
        and returns the actual gain setting and tilt setting values of EDFA3.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA3(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())

## EDFA 4
class custom_set_EDFA_4(BaseTool):
    name = "set_EDFA_4"
    description = '''
        Args: a float value representing the gain set for EDFA4.a float value representing the tilt set for EDFA4.
        Returns: The actual gain setting and tilt setting values of EDFA4.
        Note: Set the gain and tilt for EDFA4
        and returns the actual gain setting and tilt setting values of EDFA4.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA4(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())
    
## EDFA 5
class custom_set_EDFA_5(BaseTool):
    name = "set_EDFA_5"
    description = '''
        Args: a float value representing the gain set for EDFA5.a float value representing the tilt set for EDFA5.
        Returns: The actual gain setting and tilt setting values of EDFA5.
        Note: Set the gain and tilt for EDFA5
        and returns the actual gain setting and tilt setting values of EDFA5.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA5(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())

## EDFA 6
class custom_set_EDFA_6(BaseTool):
    name = "set_EDFA_6"
    description = '''
        Args: a float value representing the gain set for EDFA6.a float value representing the tilt set for EDFA6.
        Returns: The actual gain setting and tilt setting values of EDFA6.
        Note: Set the gain and tilt for EDFA6
        and returns the actual gain setting and tilt setting values of EDFA6.
        '''
    args_schema: Type[BaseModel] = set_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gain: float, 
        tilt: float, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFA6(gain, tilt)

    async def _arun(self, 
            gain: float, 
            tilt: float, 
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gain, tilt, run_manager=run_manager.get_sync())

# %%
###### EDFA整体调整
## EDFAs' Input
class set_all_EDFA_Input(BaseModel):
    gains: list = Field(description="a list of the gain values set for the six EDFAs")
    tilts: list = Field(description="a list of the tilt values set for the six EDFAs")
    
## EDFAs
class custom_set_all_EDFA(BaseTool):
    name = "set_all_EDFA"
    description = '''
        Args: Six float values representing the gains set for six EDFAs respectively. Six float values representing the tilts set for six EDFAs respectively.
        Returns: Six float values representing the real gains of the six EDFAs respectively. Six float values representing the real tilts of the six EDFAs respectively.
        Note: Set the gain and tilt for EDFA1 and returns the actual gain setting and tilt setting values of EDFA1.
        '''
    args_schema: Type[BaseModel] = set_all_EDFA_Input
    return_direct: bool = False

    def _run(
        self, 
        gains: list, 
        tilts: list, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        return setEDFAs(gains, tilts)

    async def _arun(self, 
        gains: list, 
        tilts: list, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(gains, tilts, run_manager=run_manager.get_sync())

# %%
###### EDFA input/ouput power query
## Input
class query_power_index(BaseModel):
    index: int = Field(description="the index of the queried EDFA")
    
class custom_query_input_power(BaseTool):
    name = "query_input_power"
    description = '''
        Returns: One float value in dBm representing the input power of the queried EDFA.
        Note: Query the input power of the queried EDFA
        '''
    args_schema: Type[BaseModel] = query_power_index
    return_direct: bool = False

    def _run(
        self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        '''use the tool'''
        return input_power(index)

    async def _arun(self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(index, run_manager=run_manager.get_sync())

class custom_query_output_power(BaseTool):
    name = "query_output_power"
    description = '''
        Returns: One float value in dBm representing the output power of the queried EDFA.
        Note: Query the output power of the queried EDFA
        '''
    args_schema: Type[BaseModel] = query_power_index
    return_direct: bool = False

    def _run(
        self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        '''use the tool'''
        return output_power(index)

    async def _arun(self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(index, run_manager=run_manager.get_sync())

# %%
###### Add and drop wavelengths
## Input
class wavelength_index(BaseModel):
    index: int = Field(description="the index of the batch of wavelengths")
    
class custom_add_wavebatch(BaseTool):
    name = "add_wavebatch"
    description = '''
        Returns: One bool value respresnting whether the wavelength batch adding operation is successful or not
        Note: Add the wavelength batch with the specific index
        '''
    args_schema: Type[BaseModel] = wavelength_index
    return_direct: bool = False

    def _run(
        self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        '''use the tool'''
        return add_wavelength_batch(index)

    async def _arun(self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(index, run_manager=run_manager.get_sync())

class custom_drop_wavebatch(BaseTool):
    name = "drop_wavebatch"
    description = '''
        Returns: One bool value respresnting whether the wavelength batch droping operation is successful or not
        Note: Drop the wavelength batch with the specific index
        '''
    args_schema: Type[BaseModel] = wavelength_index
    return_direct: bool = False

    def _run(
        self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        '''use the tool'''
        return drop_wavelength_batch(index)

    async def _arun(self,
        index: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(index, run_manager=run_manager.get_sync())
    

class custom_check_osc(BaseTool):
    name = "check_osc"
    description = '''
        Returns: One str of the OSC message.
        Note: Check and return the OSC message.
        '''
    return_direct: bool = False

    def _run(
        self,
        tool_input: str='',
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    )->str:
        '''use the tool'''
        return check_osc_b15()
    async def _arun(self,
                    tool_input: str='',
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                    **kwargs
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(run_manager=run_manager.get_sync())
    

# %% main
if __name__ == '__main__':
    # autoDT = custom_autoDTwave()
    # print(autoDT.run(''))
    # query_Q=custom_query_Q()

    ## 验证
    # print(query_Q.run(''))


    set_EDFA_1 = custom_set_EDFA_1()
    set_EDFA_2 = custom_set_EDFA_2()
    set_EDFA_3 = custom_set_EDFA_3()
    set_EDFA_4 = custom_set_EDFA_4()
    set_EDFA_5 = custom_set_EDFA_5()
    set_EDFA_6 = custom_set_EDFA_6()
    # co = custom_check_osc()
    

    ## 验证
    # print(co.run(''))
    # print(set_EDFA_6.run({"gain":20.2, "tilt":-0.2}))    

    # query_input_power=custom_query_input_power()
    # query_output_power=custom_query_output_power()
    ## 验证
    # print(query_output_power.run({"index":6}))
    
    add_wavebatch=custom_add_wavebatch()
    drop_wavebatch=custom_drop_wavebatch()
    # 验证
    # print(add_wavebatch.run({"index":4}))
    # print(add_wavebatch.run({"index":6}))
    # print(add_wavebatch.run({"index":1}))
    # print(add_wavebatch.run({"index":3}))
    # print(add_wavebatch.run({"index":2}))
    # # # # print(drop_wavebatch.run({"index":3}))
    # # # print(drop_wavebatch.run({"index":4}))
    # print(add_wavebatch.run({"index":3}))
    # print(add_wavebatch.run({"index":4}))
    print(add_wavebatch.run({"index":6}))
    # print(drop_wavebatch.run({"index":4}))
    # print(drop_wavebatch.run({"index":3}))
    # print(drop_wavebatch.run({"index":4}))
    # print(drop_wavebatch.run({"index":5}))
    # print(drop_wavebatch.run({"index":6}))
    # print(add_wavebatch.run({"index":6}))
    # print(add_wavebatch.run({"index":1}))

    
    
    # set_all_EDFA = custom_set_all_EDFA()
    # input_data = {"gains": [18.2, 27.2, 16.2, 10.2, 26.2, 20.2], "tilts": [-0.2, -0.2, -1.2, -1.2, -1.2, -0.2]}
    # input_data = {"gains": [16.5, 28.9, 20.5,  8.1, 25.4,18.1,], "tilts": [-1.2, -0.8, -1.1, -0.1, -1.5, -1. ]}
# 
    # response = set_all_EDFA._run(**input_data)
    # print(response)
    # set_all_EDFA.run(**{"gains":[18.2, 27.2, 16.2, 10.2, 26.2, 20.2], "tilts":[0.2, 0.2, -1.2, -1.2, -1.2, -0.2]})
    
    
    
    # import numpy as np
    # from bayes_opt import BayesianOptimization
    # from utilities.utils import setEDFAs
    # from bayes_opt.util import UtilityFunction
    # from Raman_simulator_for_LLM.gnpy.tools.GlobalControl import GlobalControl
    # logname = 'opt_BO_edfa'+datetime.now().strftime("%Y%m%d-%H%M%S")
    # GlobalControl.init_logger(logname, 1, 'modified',
    #                         file_output_dir=r'logs/'+logname+'.log')
    # logger = GlobalControl.logger
    # logger.debug('All packages are imported. Logger is initialized.')
    # # 假设的目标函数（你需要用实际的函数替换这个）
    # def target_function(x0, x1, x2, x3, x4, x5):
    #     retrungain,returntilts =setEDFAs(gains = [x0, x1, x2, x3, x4, x5],tilts = None)
    #     res = query_Q.run('')
    #     logger.info(f"Q—average:{np.mean(np.array(res))}")
    #     logger.info(f"gain:{(np.array(retrungain))}")
    #     logger.info(f"tilt:{(np.array(returntilts))}")
    #     return np.mean(res)


    # # 定义每个维度的搜索范围
    # # x_max = np.array([1,1,1,1,1,1,23,30,22,15,30,22,0,0,0,0,0,0 ])
    # # x_min = np.array([0,0,0,0,0,0,14,25,16,8,25,16,-2,-2,-2,-2,-2,-2])
    # # qot_test()
    # min_vals = np.array([14,25,16,8,25,16])
    # max_vals = np.array([23,30,22,15,30,22])

    # pbounds = {"x%d" % i: (min_vals[i], max_vals[i]) for i in range(len(min_vals))}
    # initial_points = [
    # {"x0": 16, "x1": 25, "x2": 17, "x3": 9, "x4": 25, "x5": 18},        
    # ]
    # optimizer = BayesianOptimization(
    #         f=target_function,
    #         pbounds=pbounds,  # 设置每个维度的搜索范围
    #     )
    # for point in initial_points:
    #     target_value = target_function(**point)
    #     optimizer.register(params=point, target=target_value)
    # # 创建一个贝叶斯优化对象


    # # 运行优化
    # optimizer.maximize(
    #     init_points=1,
    #     n_iter=20,
    # )
        
    