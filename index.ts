import * as tf from '@tensorflow/tfjs'
import { SymbolicTensor } from "@tensorflow/tfjs-layers/src/engine/topology";

/**
 * 1. 가르칠 데이터를 마련한다
 */
const 온도 = [20,21,22,23]
const 판매량 = [40,42,44,46]
const 원인 = tf.tensor(온도)
const 결과 = tf.tensor(판매량)

/**
 * 2. 모델의 모양을 만든다
 */
const inputs = tf.input({
  // 입력이 1개라는 뜻
  shape: [1]
})
const outputs  = tf.layers.dense({
  // 출력이 1개라는 뜻
  units: 1
}).apply(inputs) as tf.SymbolicTensor[]
const model = tf.model({ inputs, outputs })
const compileParam: tf.ModelCompileArgs = {
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError
}
model.compile(compileParam)

/**
 * 3. 데이터로 모델을 학습시킨다
 */
// 학습 횟수는 100번
const fitParam: tf.ModelFitArgs = { epochs: 100 }
// const fitParam: tf.ModelFitArgs = {
//   epochs: 100,
//   callbacks: {
//     onEpochEnd(epoch, logs) {
//       console.log(epoch, logs);
//     }
//   }
// }
model.fit(원인, 결과, fitParam).then(result => {
  // 4. 모델을 이용합니다.
  // 4.1 기존의 데이터를 이용
  const predicts = model.predict(원인);

  console.log(predicts.toString());
})

/**
 * 4. 새로운 데이터를 이용한다
 */
const 다음주온도 = [15, 16, 17, 18, 19]
const 다음주원인 = tf.tensor(다음주온도)
const 다음주결과 = model.predict(다음주원인)
다음주원인.print()
