/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
  {
    title: 'What\'s new: Survey-weighted DiD',
    Svg: require('../../static/img/fontawesome/chart-line.svg').default,
    description: (
      <>
        balance now pairs with{' '}
        <a href="https://github.com/igerber/diff-diff">diff-diff</a> for
        modern Difference-in-Differences (Callaway-Sant'Anna,
        Sun-Abraham, BJS) with built-in survey-design variance. Install
        with <code>pip install "balance[did]"</code> and see the{' '}
        <a href="https://import-balance.org/docs/tutorials/balance_diff_diff_brfss/">
          BRFSS DiD tutorial
        </a>{' '}
        for the end-to-end workflow.
        {/* NOTE: this link is INTENTIONALLY absolute (mirrors the
            ``tutorials/index.mdx`` strategy). The tutorial notebook
            ships in a separate diff in this stack, so a relative
            ``/docs/tutorials/balance_diff_diff_brfss`` would 404 in the
            per-diff ephemeral build until that diff lands. Absolute
            external URLs bypass Docusaurus runtime resolution. */}
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--3')}>
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
