/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This software may be used and distributed according to the terms of the
 * GNU General Public License version 2.
 */

import React, { useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

export default function HTMLLoader(props) {
  let src = props.docFile;
  const resize = (frame) => {
    const doc = frame.contentWindow.document;
    frame.height = frame.contentWindow.document.body.scrollHeight + 'px';
  };
  const onLoad = (e) => {
    const frame = e.target;
    const doc = frame.contentWindow.document;
    const observer = new MutationObserver((list, obj) => { resize(frame); });
    observer.observe(doc.body, {attributes:true, childList:true, subtree: true});
    resize(frame);
  };
  const frameRef = useRef(null)
  const f = <iframe
    ref={frameRef}
    src={src}
    style={{
      marginTop: '0',
      marginLeft: '0',
      marginRight: '0',
      minHeight: '900px'

  }}
    frameBorder="0"
    width="100%"
    scrolling="no"
    onLoad={onLoad} />;
  return f;
};
