import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-node'
import path from 'path'

(async () => {
  try {
    /**
     * 1. 가르칠 데이터를 마련한다
     */
    const 온도 = [ 20, 21, 22, 23 ]
    const 판매량 = [ 40, 42, 44, 46 ]
    const 원인 = tf.tensor(온도)
    const 결과 = tf.tensor(판매량)

    /**
     * 2. 모델의 모양을 만든다
     */
    const inputs = tf.input({
      // 입력이 1개라는 뜻
      shape: [ 1 ]
    })
    const outputs = tf.layers.dense({
      // 출력이 1개라는 뜻
      units: 1
    }).apply(inputs) as tf.SymbolicTensor[]
    const model = tf.model({inputs, outputs})
    const compileParam = {
      optimizer: tf.train.adam(),
      // 평균 제곱 오차를 손실함수로 사용한다
      loss: tf.losses.meanSquaredError
    }
    model.compile(compileParam as unknown as tf.ModelCompileArgs)

    /**
     * 3. 데이터로 모델을 학습시킨다
     */
    const fitParam: tf.ModelFitArgs = {
      // 학습 횟수는 10000번
      epochs: 10000,
      callbacks: {
        onEpochEnd (epoch: number, logs: tf.Logs = {loss: 0}) {
          // 평균 제곱 오차에 루트를 씌우면 평균 제곱근 오차가 된다.
          // 평균 제곱근 오차가 0에 가까워질수록 학습이 잘 된 것으로 보면 된다.
          console.log(epoch, logs, 'RMSE => ', Math.sqrt(logs.loss));
        }
      }
    }
    await model.fit(원인 as any, 결과 as any, fitParam)

    // 4. 모델을 이용한다.
    // 4.1 기존의 데이터를 이용
    const predicts = model.predict(원인 as any);

    // 5. 모델의 실체는 입력값 * 가중치 + bias로 나타나는 1차함수
    // @ts-ignore
    const weight = model.getWeights()[0].arraySync()[0][0]
    // @ts-ignore
    const bias = model.getWeights()[1].arraySync()[0]
    console.log('가중치 =>', weight);
    console.log('bias =>', bias);
    console.log(predicts.toString());

    // 6. 모델을 저장한다
    await model.save(`file://${path.join(process.cwd(), 'hello-tsjs-model')}`)

    console.log('종료');
  } catch (e) {
    console.log(e);
  }

// /**
//  * 4. 새로운 데이터를 이용한다
//  */k
// const 다음주온도 = [15, 16, 17, 18, 19]
// const 다음주원인 = tf.tensor(다음주온도)
// const 다음주결과 = model.predict(다음주원인)
// 다음주원인.print()
})()

