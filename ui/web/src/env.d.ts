/// <reference types="vite/client" />
/// <reference types="element-plus/global" />
import type { Directive } from "vue";

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<object, object, unknown>;
  export default component;
}

declare module "vue" {
  interface GlobalDirectives {
    loading: Directive;
  }
}
