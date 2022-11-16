/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This software may be used and distributed according to the terms of the
 * GNU General Public License version 2.
 */

import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Easy to Use',
    Svg: require('../../static/img/fontawesome/users.svg').default,
    description: (
      <>
        Provides a simple workflow, aiming to empower researcher with minimal background in Python or programming.
      </>
    ),
  },
  {
    title: 'End-to-End Workflow',
    Svg: require('../../static/img/fontawesome/layer-group.svg').default,
    description: (
      <>
        Provides a full workflow: from understanding the biases in the data,
        producing weights to balance the data, evaluating the quality of the weights,
        and producing weighted estimations.
      </>
    ),
  },
  {
    title: 'Open Source Python Software',
    Svg: require('../../static/img/fontawesome/code.svg').default,
    description: (
      <>
        The balance package is (one of a handful of) open-source survey statistics software written in Python.
        It leverages Python's advantages of being open sourced, well-supported,
        easy to learn and flexible environment, which is used for production systems in the industry and academic research.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
