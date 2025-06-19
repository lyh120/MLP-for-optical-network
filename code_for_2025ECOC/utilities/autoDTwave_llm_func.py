import sys
import os

# 将上层目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys

from utilities.utils import *
from auto_dt_wave.autoDT_3wave_146 import autoDTwave_3wave

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

class custom_autoDTwave(BaseTool):
    name = "Optimization tool"
    description = '''
        Returns: The the final Q-factors after the optimization.
        Note: Optimize the network.
        '''
    return_direct: bool = False

    def _run(
        self,
        tool_input: str='',
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    )->str:
        '''use the tool'''
        return autoDTwave_3wave()
    async def _arun(self,
                    tool_input: str='',
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                    **kwargs
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(run_manager=run_manager.get_sync())

# %% main
if __name__ == '__main__':
    autoDT = custom_autoDTwave()
    print(autoDT.run(''))