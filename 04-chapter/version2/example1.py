from langchain_core.runnables import RunnableLambda


if __name__ == "__main__":
    
    # 使用 `|` 运算符构造的 RunnableSequence
    sequence = RunnableLambda(lambda x: x - 1) | RunnableLambda(lambda x: x * 2)
    print(sequence.invoke(3)) # 4
    sequence.ainvoke
    print(sequence.batch([1, 2, 3])) # [0, 2, 4]
    # 包含使用字典字面值构造的 RunnableParallel 的序列
    sequence = RunnableLambda(lambda x: x * 2) | {
        'sub_1': RunnableLambda(lambda x: x - 1),
        'sub_2': RunnableLambda(lambda x: x - 2)
    }
    print(sequence.invoke(3)) # {'sub_1': 5, 'sub_2': 4}