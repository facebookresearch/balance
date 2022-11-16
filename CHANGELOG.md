**WORK-IN-PROGRESS. FILL IN BEFORE PUBLIC RELEASE!**

16.5.2 (September 18, 2018)

### React DOM

* Fixed a recent `<iframe>` regression ([@JSteunou](https://github.com/JSteunou) in [#13650](https://github.com/facebook/react/pull/13650))
* Fix `updateWrapper` so that `<textarea>`s no longer re-render when data is unchanged ([@joelbarbosa](https://github.com/joelbarbosa) in [#13643](https://github.com/facebook/react/pull/13643))

### Schedule (Experimental)

* Renaming "tracking" API to "tracing" ([@bvaughn](https://github.com/bvaughn) in [#13641](https://github.com/facebook/react/pull/13641))
* Add UMD production+profiling entry points ([@bvaughn](https://github.com/bvaughn) in [#13642](https://github.com/facebook/react/pull/13642))

16.5.1 (September 13, 2018)

### React

* Improve the warning when `React.forwardRef` receives an unexpected number of arguments. ([@andresroberto](https://github.com/andresroberto) in [#13636](https://github.com/facebook/react/issues/13636))

### React DOM

* Fix a regression in unstable exports used by React Native Web. ([@aweary](https://github.com/aweary) in [#13598](https://github.com/facebook/react/issues/13598))
