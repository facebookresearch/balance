/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const {fbContent} = require('docusaurus-plugin-internaldocs-fb/internal');
const math = require('remark-math');
const katex = require('rehype-katex');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'balance',
  tagline: 'A python package for balancing biased data samples',
  // TODO[scubasteve]: Migrate to final URL once set up
  url: 'https://internalfb.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  trailingSlash: true,
  favicon: 'img/balance_logo/icon.png',
  organizationName: 'facebook',
  projectName: 'balance',

  presets: [
    [
      require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
      /** @type {import('docusaurus-plugin-internaldocs-fb').PresetOptions} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: fbContent({
            internal: 'https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/core_stats/balance/parent_balance/website',
            external: 'https://github.com/facebookresearch/balance/tree/main/website',
          }),
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        staticDocsProject: 'Balance',
        trackingFile: 'fbcode/core_stats/balance/WATCHED_FILES',
        'remark-code-snippets': {
          baseDir: '..',
        },
        enableEditor: true,
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'balance',
        logo: {
          alt: 'balance Logo',
          src: 'img/balance_logo/icon.svg',
        },
        items: [
          {to: 'blog', label: 'Blog', position: 'right'},
          {
            type: 'doc',
            docId: 'docs/overview',
            position: 'right',
            label: 'Docs',
          },
          {
            type: 'doc',
            docId: 'tutorials/index',
            position: 'right',
            label: 'Tutorials',
          },
          {
            type: 'doc',
            docId: 'api_reference/index',
            position: 'right',
            label: 'API Reference',
          },
          {
            type: 'doc',
            docId: 'docs/changelog',
            position: 'right',
            label: 'Changelog',
          },
          // Please keep GitHub link to the right for consistency.
          {
            href: 'https://github.com/facebookresearch/balance',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Legal',
            // Please do not remove the privacy and terms, it's a legal requirement.
            items: [
              {
                label: 'Privacy',
                href: 'https://opensource.fb.com/legal/privacy/',
              },
              {
                label: 'Terms',
                href: 'https://opensource.fb.com/legal/terms/',
              },
              {
                label: 'Data Policy',
                href: 'https://opensource.fb.com/legal/data-policy/',
              },
              {
                label: 'Cookie Policy',
                href: 'https://opensource.fb.com/legal/cookie-policy/',
              },
            ],
          },
        ],
        logo: {
          alt: 'Meta Open Source Logo',
          // This default includes a positive & negative version, allowing for
          // appropriate use depending on your site's style.
          src: '/img/meta_opensource_logo_negative.svg',
          href: 'https://opensource.fb.com',
        },
        copyright: `
        Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.<br />
        Documentation Content Licensed Under <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY-4.0</a>.<br />`
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
      colorMode: {
        disableSwitch: true,
      },
    }),
});
