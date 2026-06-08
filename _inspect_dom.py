from playwright.sync_api import sync_playwright
import time, json

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    pg.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
    time.sleep(6)

    js = """() => {
      const out=[];
      for (const el of document.querySelectorAll('*')){
        if (el.children.length===0 && el.textContent && el.textContent.includes('keyboard_double')){
          out.push({tag:el.tagName, cls:String(el.className), testid:el.getAttribute('data-testid'),
                    text:el.textContent.trim().slice(0,40),
                    p_testid: el.parentElement && el.parentElement.getAttribute('data-testid'),
                    p_cls: el.parentElement && String(el.parentElement.className),
                    gp_testid: el.parentElement && el.parentElement.parentElement && el.parentElement.parentElement.getAttribute('data-testid')});
        }
      }
      return out;
    }"""
    print("=== EXPANDED: collapse-icon elements ===")
    print(json.dumps(pg.evaluate(js), indent=2))

    # Now collapse the sidebar and inspect the EXPAND control
    try:
        pg.evaluate("""() => {
          const btns=[...document.querySelectorAll('button')];
          const c=btns.find(b=>b.textContent && b.textContent.includes('keyboard_double'));
          if(c) c.click();
        }""")
        time.sleep(3)
        print("=== COLLAPSED: expand-icon elements ===")
        print(json.dumps(pg.evaluate(js), indent=2))
    except Exception as e:
        print("collapse step err:", e)

    b.close()
